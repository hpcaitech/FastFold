import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fastfold.model.fastnn.kernel import mask_softmax, mask_bias_softmax
from fastfold.model.fastnn.kernel import LayerNorm

from .initializer import glorot_uniform_af

from fastfold.model.fastnn.kernel import bias_sigmod_ele, bias_ele_dropout_residual
from fastfold.distributed import gather, scatter
from fastfold.distributed.comm_async import gather_async, gather_async_opp


CHUNK_SIZE = None


def set_chunk_size(chunk_size):
    global CHUNK_SIZE
    CHUNK_SIZE = chunk_size


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


class ChunkTransition(nn.Module):

    def __init__(self, d, n=4):
        super(ChunkTransition, self).__init__()
        self.norm = LayerNorm(d)
        self.linear1 = Linear(d, n * d, initializer='relu')
        self.linear2 = Linear(n * d, d, initializer='zeros')

    def forward(self, src):
        para_dim = src.shape[1]
        chunk_size = CHUNK_SIZE
        if CHUNK_SIZE == None:
            chunk_size = para_dim

        out = torch.empty_like(src)
        for ax in range(0, para_dim, chunk_size):
            x = self.norm(src[:, ax:ax + chunk_size, :, :])
            x = self.linear2(F.relu(self.linear1(x)))
            out[:, ax:ax + chunk_size, :, :] = x
        out.add_(src)
        return out


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

    def forward(self, M, M_mask):
        M = self.layernormM(M)
        right_act = self.linear_b(M)

        right_act_all, work = gather_async(right_act, dim=2)
        # right_act_all = gather(right_act, dim=2)

        left_act = self.linear_a(M)
        M_mask = M_mask.unsqueeze(-1)
        M_mask_col = scatter(M_mask, dim=2)
        left_act = M_mask_col * left_act
        norm = torch.einsum('bsid,bsjd->bijd', M_mask_col, M_mask)

        right_act_all = gather_async_opp(right_act_all, work, dim=2)

        right_act_all = M_mask * right_act_all

        para_dim = left_act.shape[2]
        chunk_size = CHUNK_SIZE
        if CHUNK_SIZE == None:
            chunk_size = para_dim

        out = []
        for ax in range(0, para_dim, chunk_size):
            left_act_part = left_act[:, :, ax:ax + chunk_size, :]

            O = torch.einsum('bsid,bsje->bijde', left_act_part, right_act_all)

            O = rearrange(O, 'b i j d e -> b i j (d e)')

            out.append(self.o_linear(O))

        Z = torch.cat(out, dim=1)

        Z /= (1e-3 + norm)

        return Z


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

        if nonbatched_bias is not None:
            # logits += nonbatched_bias.unsqueeze(1)
            bias = gather_async_opp(*nonbatched_bias, dim=1)
            bias = rearrange(bias, 'b q k h -> b h q k')

        output = []
        for ax in range(0, para_dim, chunk_size):

            in_data_part = in_data[:, ax:ax + chunk_size, :, :]
            mask_part = mask[:, ax:ax + chunk_size, :]

            qkv = self.to_qkv(in_data_part).chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b1 b2 n (h d) -> b1 b2 h n d', h=self.n_head), qkv)

            q = q * self.scaling

            logits = torch.matmul(q, k.transpose(-1, -2))

            if nonbatched_bias is not None:
                weights = mask_bias_softmax(logits, mask_part, bias.unsqueeze(1))
            else:
                weights = mask_softmax(logits, mask)

            weighted_avg = torch.matmul(weights, v)
            weighted_avg = rearrange(weighted_avg, 'b1 b2 h n d -> b1 b2 n (h d)')

            if self.gating:
                gate_values = self.gating_linear(in_data_part)
                weighted_avg = bias_sigmod_ele(gate_values, self.gating_bias, weighted_avg)

            output.append(self.o_linear(weighted_avg))

        output = torch.cat(output, dim=1)
        
        return output


def permute_final_dims(tensor, inds):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


class ChunkTriangleMultiplicationOutgoing(nn.Module):

    def __init__(self, d_pair, p_drop, c=128):
        super(ChunkTriangleMultiplicationOutgoing, self).__init__()
        self.d_pair = d_pair
        self.c = c

        self.layernorm1 = LayerNorm(d_pair)
        self.left_right_projection = Linear(d_pair, 2 * c)
        self.left_right_gate = Linear(d_pair, 2 * c, initializer='zeros', bias_init=1.)

        self.output_gate = Linear(d_pair, d_pair, initializer='zeros', bias_init=1.)
        self.layernorm2 = LayerNorm(c)
        self.output_projection = Linear(d_pair, d_pair, initializer='zeros', use_bias=False)
        self.output_bias = nn.parameter.Parameter(data=torch.zeros((d_pair,)), requires_grad=True)

        self.p_drop = p_drop

    def forward(self, Z_raw, Z_mask_row):

        para_dim = Z_raw.shape[1]
        chunk_size = 256
        if CHUNK_SIZE == None:
            chunk_size = para_dim

        output = torch.empty_like(Z_raw)
        
        for i in range(0, para_dim, chunk_size):
            zi = Z_raw[:, i:i + chunk_size, :, :]
            zi = self.layernorm1(zi)
            gi = torch.sigmoid(self.left_right_gate(zi))
            i_left_right_proj_act = self.left_right_projection(zi)
            i_left_right_proj_act = Z_mask_row[:, i:i + chunk_size, :].unsqueeze(-1) * i_left_right_proj_act
            i_left_right_proj_act *= gi
            left_proj_act, _ = i_left_right_proj_act.chunk(2, dim=-1)
            left_proj_act = permute_final_dims(left_proj_act, (2, 0, 1))
            
            for j in range(0, para_dim, chunk_size):
                                
                zj = Z_raw[:, j:j + chunk_size, :, :]
                zj = self.layernorm1(zj)
                gj = torch.sigmoid(self.left_right_gate(zj))
                j_left_right_proj_act = self.left_right_projection(zj)
                j_left_right_proj_act = Z_mask_row[:, j:j + chunk_size, :].unsqueeze(-1) * j_left_right_proj_act
                j_left_right_proj_act *= gj
                _, right_proj_act = j_left_right_proj_act.chunk(2, dim=-1)
                
                p = torch.matmul(
                    left_proj_act,
                    permute_final_dims(right_proj_act, (2, 1, 0)),
                )
                p = permute_final_dims(p, (1, 2, 0))
                output[:, i:i + chunk_size, j:j + chunk_size, :] = p

        dropout_mask = torch.ones_like(Z_raw[:, 0:1, :, :], device=Z_raw.device, dtype=Z_raw.dtype)
        for i in range(0, para_dim, chunk_size):
            z_raw = output[:, i:i + chunk_size, :, :]
            g = torch.sigmoid(self.output_gate(z_raw))
            z = self.output_projection(self.layernorm2(z_raw))
            z = bias_ele_dropout_residual(z,
                                    self.output_bias,
                                    g,
                                    dropout_mask,
                                    z_raw,
                                    prob=self.p_drop,
                                    training=self.training)
            output[:, i:i + chunk_size, :, :] = z
        return output


class ChunkTriangleMultiplicationIncoming(nn.Module):

    def __init__(self, d_pair, p_drop, c=128):
        super(ChunkTriangleMultiplicationIncoming, self).__init__()
        self.d_pair = d_pair
        self.c = c

        self.layernorm1 = LayerNorm(d_pair)
        self.left_right_projection = Linear(d_pair, 2 * c)
        self.left_right_gate = Linear(d_pair, 2 * c, initializer='zeros', bias_init=1.)

        self.output_gate = Linear(d_pair, d_pair, initializer='zeros', bias_init=1.)
        self.layernorm2 = LayerNorm(c)
        self.output_projection = Linear(d_pair, d_pair, initializer='zeros', use_bias=False)
        self.output_bias = nn.parameter.Parameter(data=torch.zeros((d_pair,)), requires_grad=True)

        self.p_drop = p_drop

    def forward(self, Z_raw, Z_mask_col):

        para_dim = Z_raw.shape[1]
        chunk_size = 256
        if CHUNK_SIZE == None:
            chunk_size = para_dim

        output = torch.empty_like(Z_raw)
        
        for i in range(0, para_dim, chunk_size):
            zi = Z_raw[:, :, i:i + chunk_size, :]
            zi = self.layernorm1(zi)
            gi = torch.sigmoid(self.left_right_gate(zi))
            i_left_right_proj_act = self.left_right_projection(zi)
            i_left_right_proj_act = Z_mask_col[:, :, i:i + chunk_size].unsqueeze(-1) * i_left_right_proj_act
            i_left_right_proj_act *= gi
            left_proj_act, _ = i_left_right_proj_act.chunk(2, dim=-1)
            left_proj_act = permute_final_dims(left_proj_act, (2, 1, 0))
            
            for j in range(0, para_dim, chunk_size):
                                
                zj = Z_raw[:, :, j:j + chunk_size, :]
                zj = self.layernorm1(zj)
                gj = torch.sigmoid(self.left_right_gate(zj))
                j_left_right_proj_act = self.left_right_projection(zj)
                j_left_right_proj_act = Z_mask_col[:, :, j:j + chunk_size].unsqueeze(-1) * j_left_right_proj_act
                j_left_right_proj_act *= gj
                _, right_proj_act = j_left_right_proj_act.chunk(2, dim=-1)

                p = torch.matmul(
                    left_proj_act,
                    permute_final_dims(right_proj_act, (2, 0, 1)),
                )
                p = permute_final_dims(p, (1, 2, 0))
                output[:, i:i + chunk_size, j:j + chunk_size, :] = p

        dropout_mask = torch.ones_like(Z_raw[:, 0:1, :, :], device=Z_raw.device, dtype=Z_raw.dtype)
        for i in range(0, para_dim, chunk_size):
            z_raw = output[:, i:i + chunk_size, :, :]
            g = torch.sigmoid(self.output_gate(z_raw))
            z = self.output_projection(self.layernorm2(z_raw))
            z = bias_ele_dropout_residual(z,
                                    self.output_bias,
                                    g,
                                    dropout_mask,
                                    z_raw,
                                    prob=self.p_drop,
                                    training=self.training)
            output[:, i:i + chunk_size, :, :] = z
        return output
