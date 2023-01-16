import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn import LayerNorm

from fastfold.habana.distributed import gather, scatter

from .initializer import glorot_uniform_af
from .kernel import bias_sigmod_ele

from fastfold.habana.distributed import gather, scatter
from fastfold.habana.fastnn.custom_op import fused_softmax, fused_softmax_bias

CHUNK_SIZE = None
DEBUG = False


def set_chunk_size(chunk_size):
    global CHUNK_SIZE
    CHUNK_SIZE = chunk_size


def get_chunk_size():
    global CHUNK_SIZE
    return CHUNK_SIZE


class DropoutRowwise(nn.Module):

    def __init__(self, p):
        super(DropoutRowwise, self).__init__()
        self.p = p
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        dropout_mask = torch.ones_like(x[:, 0:1, :, :])
        dropout_mask = self.dropout(dropout_mask)
        return dropout_mask * x


class DropoutColumnwise(nn.Module):

    def __init__(self, p):
        super(DropoutColumnwise, self).__init__()
        self.p = p
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        dropout_mask = torch.ones_like(x[:, :, 0:1, :])
        dropout_mask = self.dropout(dropout_mask)
        return dropout_mask * x


class Transition(nn.Module):

    def __init__(self, d, n=4):
        super(Transition, self).__init__()
        self.norm = LayerNorm(d)
        self.linear1 = Linear(d, n * d, initializer='relu')
        self.linear2 = Linear(n * d, d, initializer='zeros')

    def forward(self, src):
        x = self.norm(src)
        x = self.linear2(F.relu(self.linear1(x)))
        return src + x


class OutProductMean(nn.Module):

    def __init__(self, n_feat=64, n_feat_out=128, n_feat_proj=32):
        super(OutProductMean, self).__init__()

        self.layernormM = LayerNorm(n_feat)
        self.linear_a = Linear(n_feat, n_feat_proj)
        self.linear_b = Linear(n_feat, n_feat_proj)

        self.o_linear = Linear(n_feat_proj * n_feat_proj,
                               n_feat_out,
                               initializer='zero',
                               use_bias=True)

    def forward(self, M, M_mask, Z_raw):
        Z = torch.empty_like(Z_raw)
        M = self.layernormM(M)
        left_act = self.linear_a(M)
        right_act = self.linear_b(M)

        right_act_all = gather(right_act, dim=2)

        M_mask = M_mask.unsqueeze(-1)
        M_mask_col = scatter(M_mask, dim=2)
        left_act = M_mask_col * left_act
        right_act_all = M_mask * right_act_all

        norm = torch.einsum('...ab,...ad->...bd',
                            M_mask_col.squeeze(-1).squeeze(0),
                            M_mask.squeeze(-1).squeeze(0)).unsqueeze(-1).unsqueeze(0)

        para_dim = left_act.shape[2]
        chunk_size = CHUNK_SIZE
        if CHUNK_SIZE == None:
            chunk_size = para_dim

        out = []
        for ax in range(0, para_dim, chunk_size):
            left_act_part = left_act[:, :, ax:ax + chunk_size, :]

            # O = torch.einsum('sid,sje->ijde', left_act_part.squeeze(0), right_act_all.squeeze(0))

            # O = rearrange(O, 'i j d e -> i j (d e)')
            left_shape = left_act_part.shape
            right_shape = right_act_all.shape
            left_act_part = left_act_part.reshape(left_shape[0], left_shape[1], left_shape[2]*left_shape[3])
            right_act_all = right_act_all.reshape(right_shape[0], right_shape[1], right_shape[2]*right_shape[3])
            # O = torch.einsum('...ab,...ad->...bd', left_act_part.squeeze(0), right_act_all.squeeze(0))
            O = torch.matmul(left_act_part.squeeze(0).transpose(1, 0), right_act_all.squeeze(0))
            O = O.reshape(left_shape[2], left_shape[3], right_shape[2], right_shape[3]).transpose(-2, -3)
            O = O.reshape(O.shape[0], O.shape[1], O.shape[2]*O.shape[3])

            O = O.unsqueeze(0)

            out.append(self.o_linear(O))

        Z = torch.cat(out, dim=1)

        Z /= (1e-3 + norm)

        return Z + Z_raw


class Linear(nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just
    like torch.nn.Linear.
    Implements the initializers in 1.11.4, plus some additional ones found
    in the code.
    """

    def __init__(
        self,
        feature_in: int,
        feature_out: int,
        initializer: str = 'linear',
        use_bias: bool = True,
        bias_init: float = 0.,
    ):
        super(Linear, self).__init__(feature_in, feature_out, bias=use_bias)

        self.use_bias = use_bias
        if initializer == 'linear':
            glorot_uniform_af(self.weight, gain=1.0)
        elif initializer == 'relu':
            glorot_uniform_af(self.weight, gain=2.0)
        elif initializer == 'zeros':
            nn.init.zeros_(self.weight)
        if self.use_bias:
            with torch.no_grad():
                self.bias.fill_(bias_init)


class SelfAttention(nn.Module):
    """
    Multi-Head SelfAttention dealing with [batch_size1, batch_size2, len, dim] tensors
    """

    def __init__(self, qkv_dim, c, n_head, out_dim, gating=True, last_bias_fuse=False):
        super(SelfAttention, self).__init__()
        self.qkv_dim = qkv_dim
        self.c = c
        self.n_head = n_head
        self.out_dim = out_dim
        self.gating = gating
        self.last_bias_fuse = last_bias_fuse

        self.scaling = self.c**(-0.5)

        self.to_qkv = Linear(qkv_dim, 3 * n_head * c, initializer='linear', use_bias=False)
        # self.to_q = Linear(qkv_dim, n_head * c, initializer='linear', use_bias=False)
        # self.to_k = Linear(qkv_dim, n_head * c, initializer='linear', use_bias=False)
        # self.to_v = Linear(qkv_dim, n_head * c, initializer='linear', use_bias=False)

        if gating:
            self.gating_bias = nn.parameter.Parameter(data=torch.ones((n_head * c,)))
            self.gating_linear = Linear(qkv_dim, n_head * c, initializer='zero', use_bias=False)

        self.o_linear = Linear(n_head * c,
                               out_dim,
                               initializer='zero',
                               use_bias=(not last_bias_fuse))

    def forward(self, in_data, mask, nonbatched_bias=None):
        """
        :param in_data: [batch_size1, batch_size2, len_qkv, qkv_dim]
        :param bias: None or [batch_size1, batch_size2, n_head, len_q, len_kv]
        :param nonbatched_bias: None or [batch_size1, n_head, len_q, len_kv]
        """

        para_dim = in_data.shape[1]
        chunk_size = CHUNK_SIZE
        if CHUNK_SIZE == None:
            chunk_size = para_dim

        output = []
        for ax in range(0, para_dim, chunk_size):

            in_data_part = in_data[:, ax:ax + chunk_size, :, :]
            mask_part = mask[:, ax:ax + chunk_size, :]

            qkv = self.to_qkv(in_data_part).chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b1 b2 n (h d) -> b1 b2 h n d', h=self.n_head), qkv)

            # q = self.to_q(in_data_part)
            # k = self.to_k(in_data_part)
            # v = self.to_v(in_data_part)

            # q, k, v = map(lambda t: rearrange(t, 'b1 b2 n (h d) -> b1 b2 h n d', h=self.n_head),
            #               [q, k, v])

            q = q * self.scaling

            logits = torch.matmul(q, k.transpose(-1, -2))

            # logits += (1e9 * (mask_part - 1))[..., :, None, None, :]

            # if nonbatched_bias is not None:
            #     logits += nonbatched_bias.unsqueeze(1)
            # weights = torch.softmax(logits, dim=-1)

            mask00 = (1e9 * (mask_part - 1))[..., :, None, None, :]
            if nonbatched_bias is not None:
                weights = fused_softmax_bias(logits, mask00, nonbatched_bias.unsqueeze(1), -1)
            else:
                weights = fused_softmax(logits, mask00, -1)

            weighted_avg = torch.matmul(weights, v)
            weighted_avg = rearrange(weighted_avg, 'b1 b2 h n d -> b1 b2 n (h d)')

            if self.gating:
                gate_values = self.gating_linear(in_data_part)
                weighted_avg = bias_sigmod_ele(gate_values, self.gating_bias, weighted_avg)

            output.append(self.o_linear(weighted_avg))

        output = torch.cat(output, dim=1)

        return output


class GlobalAttention(nn.Module):
    """
    Multi-Head SelfAttention dealing with [batch_size1, batch_size2, len, dim] tensors
    """

    def __init__(self, qkv_dim, c, n_head, out_dim):
        super(GlobalAttention, self).__init__()
        self.qkv_dim = qkv_dim
        self.c = c
        self.n_head = n_head
        self.out_dim = out_dim

        self.scaling = self.c**(-0.5)

        self.eps = 1e-10
        self.inf = 1e9

        self.to_q = Linear(qkv_dim, c * self.n_head, use_bias=False)
        self.to_kv = Linear(qkv_dim, 2 * c, initializer="linear", use_bias=False)

        self.gating_bias = nn.parameter.Parameter(data=torch.ones((n_head * c,)))
        self.gating_linear = Linear(qkv_dim, n_head * c, initializer="zero", use_bias=False)

        self.o_linear = Linear(n_head * c, out_dim, initializer="zero")

    def forward(self, m, mask):

        para_dim = m.shape[1]
        chunk_size = CHUNK_SIZE
        if CHUNK_SIZE == None:
            chunk_size = para_dim

        output = []
        for ax in range(0, para_dim, chunk_size):

            m_part = m[:, ax:ax + chunk_size, :, :]
            mask_part = mask[:, ax:ax + chunk_size, :]

            q = torch.sum(m_part * mask_part.unsqueeze(-1),
                          dim=-2) / (torch.sum(mask_part, dim=-1)[..., None] + self.eps)

            q = self.to_q(q)
            q = q.view(q.shape[:-1] + (self.n_head, -1))

            k, v = self.to_kv(m_part).chunk(2, dim=-1)

            logits = torch.matmul(q, k.transpose(-1, -2))

            # logits += (1e9 * (mask_part - 1))[..., :, None, None, :]

            weights = torch.softmax(logits, dim=-1)

            weighted_avg = torch.matmul(weights, v)
            weighted_avg = rearrange(weighted_avg, "b1 b2 h d -> b1 b2 (h d)")

            gate_values = self.gating_linear(m_part)
            weighted_avg = bias_sigmod_ele(gate_values, self.gating_bias,
                                           weighted_avg.unsqueeze(-2))

            output.append(self.o_linear(weighted_avg))

        m = torch.cat(output, dim=1)

        return m
