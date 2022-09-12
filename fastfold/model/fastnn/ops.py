from typing import Tuple
# Copyright 2022 BioMap (Beijing) Intelligence Technology Limited
# Copyright 2022 HPC-AI Technology Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fastfold.model.fastnn.kernel import mask_softmax, mask_bias_softmax
from fastfold.model.fastnn.kernel import LayerNorm

from .initializer import glorot_uniform_af

from fastfold.model.fastnn.kernel import bias_sigmod_ele, bias_ele_dropout_residual, bias_dropout_add
from fastfold.distributed import gather, scatter
from fastfold.distributed.comm_async import gather_async, gather_async_opp, get_world_size, get_rank, broadcast_sync, broadcast_async, broadcast_async_opp


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
        chunk_size = 48
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

    def forward(self, M, M_mask, Z_raw):
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

        Z = torch.empty_like(Z_raw)
        for ax in range(0, para_dim, chunk_size):
            left_act_part = left_act[:, :, ax:ax + chunk_size, :]
            O = torch.einsum('bsid,bsje->bijde', left_act_part, right_act_all)
            O = rearrange(O, 'b i j d e -> b i j (d e)')
            O = self.o_linear(O)
            Z[:, ax:ax + chunk_size, :, :] = O

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
            if nonbatched_bias[-1] == -1:
                bias = nonbatched_bias[0]
            else:
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


class AsyncChunkTriangleMultiplicationOutgoing(nn.Module):

    def __init__(self, d_pair, p_drop, c=128):
        super(AsyncChunkTriangleMultiplicationOutgoing, self).__init__()
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

        if CHUNK_SIZE == None:
            Z = self.layernorm1(Z_raw)
            left_right_proj_act = self.left_right_projection(Z)
            left_right_proj_act = Z_mask_row.unsqueeze(-1) * left_right_proj_act
            left_right_proj_act *= torch.sigmoid(self.left_right_gate(Z))
            left_proj_act, right_proj_act = left_right_proj_act.chunk(2, dim=-1)
            right_proj_act, work = gather_async(right_proj_act.contiguous(), dim=1)
            g = torch.sigmoid(self.output_gate(Z))
            left_proj_act = permute_final_dims(left_proj_act, (2, 0, 1))
            right_proj_act = gather_async_opp(right_proj_act, work, dim=1)
            p = torch.matmul(left_proj_act, permute_final_dims(right_proj_act, (2, 1, 0)),)
            ab = permute_final_dims(p, (1, 2, 0))
            ab = self.output_projection(self.layernorm2(ab))
            dropout_mask = torch.ones_like(Z[:, 0:1, :, :], device=Z.device, dtype=Z.dtype)
            return bias_ele_dropout_residual(ab,
                                             self.output_bias,
                                             g,
                                             dropout_mask,
                                             Z_raw,
                                             prob=self.p_drop,
                                             training=self.training)

        para_dim = Z_raw.shape[1]
        chunk_size = CHUNK_SIZE * 32
        world_size = get_world_size()
        rank = get_rank()
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
                right_proj_act = right_proj_act.contiguous()
                
                work = None
                right_proj_act_tmp = torch.empty_like(right_proj_act)
                
                for k in range(0, world_size):
                    
                    if world_size > 1:
                        if work:
                            broadcast_async_opp(work) # collect last broadcast
                            if k != rank:
                                right_proj_act_rec = right_proj_act_tmp.clone()
                        else:  # init first broadcast
                            if k == rank:
                                broadcast_sync(k, right_proj_act, host=True)
                            else:
                                right_proj_act_tmp = broadcast_sync(k, right_proj_act, host=False)
                                right_proj_act_rec = right_proj_act_tmp.clone()
                    
                        if k + 1 != world_size: # launch next broadcast
                            if k + 1 == rank:
                                work = broadcast_async(k + 1, right_proj_act, host=True)
                            else:
                                work = broadcast_async(k + 1, right_proj_act_tmp, host=False)
                        
                    if k == rank: # broadcast self right_proj_act
                        p = torch.matmul(
                            left_proj_act,
                            permute_final_dims(right_proj_act, (2, 1, 0)),
                        )
                        p = permute_final_dims(p, (1, 2, 0))
                        j_global = para_dim * k + j
                        output[:, i:i + chunk_size, j_global:min(j_global + chunk_size, para_dim * (k + 1)), :] = p
                        
                    else:   # receive others broadcast                        
                        p = torch.matmul(
                            left_proj_act,
                            permute_final_dims(right_proj_act_rec, (2, 1, 0)),
                        )
                        p = permute_final_dims(p, (1, 2, 0))
                        j_global = para_dim * k + j
                        output[:, i:i + chunk_size, j_global:min(j_global + chunk_size, para_dim * (k + 1)), :] = p
        
        dropout_mask = torch.ones_like(Z_raw[:, 0:1, :, :], device=Z_raw.device, dtype=Z_raw.dtype)                        
        for i in range(0, Z_raw.shape[1], chunk_size):
            z_raw = Z_raw[:, i:i + chunk_size, :, :]
            g = torch.sigmoid(self.output_gate(self.layernorm1(z_raw)))
            z = output[:, i:i + chunk_size, :, :]
            z = self.output_projection(self.layernorm2(z))
            z = bias_ele_dropout_residual(z,
                                    self.output_bias,
                                    g,
                                    dropout_mask,
                                    z_raw,
                                    prob=self.p_drop,
                                    training=self.training)
            output[:, i:i + chunk_size, :, :] = z
        return output


class AsyncChunkTriangleMultiplicationIncoming(nn.Module):

    def __init__(self, d_pair, p_drop, c=128):
        super(AsyncChunkTriangleMultiplicationIncoming, self).__init__()
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

        if CHUNK_SIZE == None:
            Z = self.layernorm1(Z_raw)
            left_right_proj_act = self.left_right_projection(Z)
            left_right_proj_act = Z_mask_col.unsqueeze(-1) * left_right_proj_act
            left_right_proj_act *= torch.sigmoid(self.left_right_gate(Z))
            left_proj_act, right_proj_act = left_right_proj_act.chunk(2, dim=-1)
            left_proj_act, work = gather_async(left_proj_act.contiguous(), dim=2)
            g = torch.sigmoid(self.output_gate(Z))
            right_proj_act = permute_final_dims(right_proj_act, (2, 0, 1))
            left_proj_act = gather_async_opp(left_proj_act, work, dim=2)
            p = torch.matmul(permute_final_dims(left_proj_act, (2, 1, 0)), right_proj_act)
            ab = permute_final_dims(p, (1, 2, 0))
            ab = self.output_projection(self.layernorm2(ab))
            dropout_mask = torch.ones_like(Z[:, 0:1, :, :], device=Z.device, dtype=Z.dtype)
            return bias_ele_dropout_residual(ab,
                                            self.output_bias,
                                            g,
                                            dropout_mask,
                                            Z_raw,
                                            prob=self.p_drop,
                                            training=self.training)
                    
        para_dim = Z_raw.shape[2]
        chunk_size = CHUNK_SIZE * 32
        world_size = get_world_size()
        rank = get_rank()
        output = torch.empty_like(Z_raw)
        
        for i in range(0, para_dim, chunk_size):
            zi = Z_raw[:, :, i:i + chunk_size, :]
            zi = self.layernorm1(zi)
            gi = torch.sigmoid(self.left_right_gate(zi))
            i_left_right_proj_act = self.left_right_projection(zi)
            i_left_right_proj_act = Z_mask_col[:, :, i:i + chunk_size].unsqueeze(-1) * i_left_right_proj_act
            i_left_right_proj_act *= gi
            _, right_proj_act = i_left_right_proj_act.chunk(2, dim=-1)
            right_proj_act = permute_final_dims(right_proj_act, (2, 0, 1))
            
            for j in range(0, para_dim, chunk_size):
                                
                zj = Z_raw[:, :, j:j + chunk_size, :]
                zj = self.layernorm1(zj)
                gj = torch.sigmoid(self.left_right_gate(zj))
                j_left_right_proj_act = self.left_right_projection(zj)
                j_left_right_proj_act = Z_mask_col[:, :, j:j + chunk_size].unsqueeze(-1) * j_left_right_proj_act
                j_left_right_proj_act *= gj
                left_proj_act, _ = j_left_right_proj_act.chunk(2, dim=-1)
                left_proj_act = left_proj_act.contiguous()
                
                work = None
                left_proj_act_tmp = torch.empty_like(left_proj_act)
                
                for k in range(0, world_size):
                    
                    if world_size > 1:
                        if work:
                            broadcast_async_opp(work) # collect last broadcast
                            if k != rank:
                                left_proj_act_rec = left_proj_act_tmp.clone()
                        else:  # init first broadcast
                            if k == rank:
                                broadcast_sync(k, left_proj_act, host=True)
                            else:
                                left_proj_act_tmp = broadcast_sync(k, left_proj_act, host=False)
                                left_proj_act_rec = left_proj_act_tmp.clone()
                        
                        if k + 1 != world_size: # launch next broadcast
                            if k + 1 == rank:
                                work = broadcast_async(k + 1, left_proj_act, host=True)
                            else:
                                work = broadcast_async(k + 1, left_proj_act_tmp, host=False)
                    
                    if k == rank: # broadcast self proj_act
                        # left: [seq,chunkj,dim] => [dim,chunkj,seq]
                        # right: [seq,chunki,dim] => [dim,seq,chunki]
                        # p: [dim,chunkj,chunki] => [chunkj,chunki,dim]
                        p = torch.matmul(
                            permute_final_dims(left_proj_act, (2, 1, 0)),
                            right_proj_act
                        )
                        p = permute_final_dims(p, (1, 2, 0))
                        j_global = para_dim * k + j
                        output[:, j_global:min(j_global + chunk_size, para_dim * (k + 1)), i:i + chunk_size, :] = p
                        
                    else:   # receive others broadcast                        
                        p = torch.matmul(
                            permute_final_dims(left_proj_act_rec, (2, 1, 0)),
                            right_proj_act
                        )
                        p = permute_final_dims(p, (1, 2, 0))
                        j_global = para_dim * k + j
                        output[:, j_global:min(j_global + chunk_size, para_dim * (k + 1)), i:i + chunk_size, :] = p
        
        dropout_mask = torch.ones_like(Z_raw[:, 0:1, :, :], device=Z_raw.device, dtype=Z_raw.dtype)
        for i in range(0, Z_raw.shape[1], chunk_size):
            z_raw = Z_raw[:, i:i + chunk_size, :, :]
            g = torch.sigmoid(self.output_gate(self.layernorm1(z_raw)))
            z = output[:, i:i + chunk_size, :, :]
            z = self.output_projection(self.layernorm2(z))
            z = bias_ele_dropout_residual(z,
                                    self.output_bias,
                                    g,
                                    dropout_mask,
                                    z_raw,
                                    prob=self.p_drop,
                                    training=self.training)
            output[:, i:i + chunk_size, :, :] = z
        return output


class ChunkTriangleAttentionStartingNode(nn.Module):

    def __init__(self, d_pair, p_drop, c=32, n_head=4):
        super(ChunkTriangleAttentionStartingNode, self).__init__()
        self.d_pair = d_pair
        self.c = c
        self.n_head = n_head
        self.p_drop = p_drop

        self.layernorm1 = LayerNorm(d_pair)
        # _init_weights = torch.nn.init.normal_(torch.zeros([d_pair, n_head]),
        #                                       std=1.0 / math.sqrt(d_pair))
        # self.linear_b_weights = nn.parameter.Parameter(data=_init_weights)

        self.linear_b = Linear(d_pair, n_head, initializer='linear', use_bias=False)
        self.attention = SelfAttention(qkv_dim=d_pair,
                                       c=c,
                                       n_head=n_head,
                                       out_dim=d_pair,
                                       gating=True,
                                       last_bias_fuse=True)

        self.out_bias = nn.parameter.Parameter(data=torch.zeros((d_pair,)), requires_grad=True)

    def forward(self, Z_raw, Z_mask):
        
        if CHUNK_SIZE == None:        
            Z = self.layernorm1(Z_raw)
            b = self.linear_b(Z)
            b, work = gather_async(b, dim=1)
            Z = self.attention(Z, Z_mask, (b, work))
            dropout_mask = torch.ones_like(Z[:, 0:1, :, :], device=Z.device, dtype=Z.dtype)
            return bias_dropout_add(Z,
                                    self.out_bias,
                                    dropout_mask,
                                    Z_raw,
                                    prob=self.p_drop,
                                    training=self.training)
        
        chunk_size = CHUNK_SIZE
        para_dim = Z_raw.shape[1]
        # z is big, but b is small. So we compute z in chunk to get b, and recompute z in chunk later instead of storing it
        b = torch.empty((Z_raw.shape[0], Z_raw.shape[1], Z_raw.shape[2], self.n_head), device=Z_raw.device, dtype=Z_raw.dtype)
        for i in range(0, para_dim, chunk_size):
            z = self.layernorm1(Z_raw[:, i:i + chunk_size, :, :])
            b[:, i:i + chunk_size, :, :] = self.linear_b(z)
        b, work = gather_async(b, dim=1)
        b = gather_async_opp(b, work, dim=1)
        b = rearrange(b, 'b q k h -> b h q k')
        
        output = torch.empty_like(Z_raw)
        dropout_mask = torch.ones_like(z[:, 0:1, :, :], device=z.device, dtype=z.dtype)
        for i in range(0, para_dim, chunk_size):
            z_raw = Z_raw[:, i:i + chunk_size, :, :]
            z = self.layernorm1(z_raw)
            z_mask = Z_mask[:, i:i + chunk_size, :]
            
            z = self.attention(z, z_mask, (b, -1))
            z =  bias_dropout_add(z,
                                    self.out_bias,
                                    dropout_mask,
                                    z_raw,
                                    prob=self.p_drop,
                                    training=self.training)
            output[:, i:i + chunk_size, :, :] = z
        
        return output


class ChunkTriangleAttentionEndingNode(nn.Module):

    def __init__(self, d_pair, p_drop, c=32, n_head=4):
        super(ChunkTriangleAttentionEndingNode, self).__init__()
        self.d_pair = d_pair
        self.c = c
        self.n_head = n_head
        self.p_drop = p_drop

        self.layernorm1 = LayerNorm(d_pair)
        self.linear_b = Linear(d_pair, n_head, initializer='linear', use_bias=False)
        self.attention = SelfAttention(qkv_dim=d_pair,
                                       c=c,
                                       n_head=n_head,
                                       out_dim=d_pair,
                                       gating=True,
                                       last_bias_fuse=True)
        self.out_bias = nn.parameter.Parameter(data=torch.zeros((d_pair,)), requires_grad=True)

    def forward(self, Z_raw, Z_mask):
        
        if CHUNK_SIZE == None:  
            Z = Z_raw.transpose(-2, -3)
            Z_mask = Z_mask.transpose(-1, -2)

            Z = self.layernorm1(Z)
            b = self.linear_b(Z)
            b, work = gather_async(b, dim=1)
            Z = self.attention(Z, Z_mask, (b, work))
            Z = Z.transpose(-2, -3)
            dropout_mask = torch.ones_like(Z[:, :, 0:1, :], device=Z.device, dtype=Z.dtype)
            return bias_dropout_add(Z,
                                    self.out_bias,
                                    dropout_mask,
                                    Z_raw,
                                    prob=self.p_drop,
                                    training=self.training)

        para_dim = Z_raw.shape[2]
        chunk_size = CHUNK_SIZE
        # z is big, but b is small. So we compute z in chunk to get b, and recompute z in chunk later instead of storing it
        b = torch.empty((Z_raw.shape[0], Z_raw.shape[2], Z_raw.shape[1], self.n_head), device=Z_raw.device, dtype=Z_raw.dtype)
        for i in range(0, para_dim, chunk_size):
            z = Z_raw[:, :, i:i + chunk_size, :].transpose(-2, -3)
            z = self.layernorm1(z)
            b[:, i:i + chunk_size, :, :] = self.linear_b(z)
        b, work = gather_async(b, dim=1)
        b = gather_async_opp(b, work, dim=1)
        b = rearrange(b, 'b q k h -> b h q k')
        
        output = torch.empty_like(Z_raw)
        dropout_mask = torch.ones_like(Z_raw[:, :, 0:1, :], device=z.device, dtype=z.dtype)
        for i in range(0, para_dim, chunk_size):
            z_raw = Z_raw[:, :, i:i + chunk_size, :]
            z = self.layernorm1(z_raw.transpose(-2, -3))
            z_mask = Z_mask[:, :, i:i + chunk_size].transpose(-1, -2)

            z = self.attention(z, z_mask, (b, -1)).transpose(-2, -3)
            z =  bias_dropout_add(z,
                                    self.out_bias,
                                    dropout_mask,
                                    z_raw,
                                    prob=self.p_drop,
                                    training=self.training)
            output[:, :, i:i + chunk_size, :] = z
        
        return output


class RecyclingEmbedder(nn.Module):
    """
    Embeds the output of an iteration of the model for recycling.

    Implements Algorithm 32.
    """

    def __init__(
        self,
        c_m: int,
        c_z: int,
        min_bin: float,
        max_bin: float,
        no_bins: int,
        inf: float = 1e8,
        **kwargs,
    ):
        """
        Args:
            c_m:
                MSA channel dimension
            c_z:
                Pair embedding channel dimension
            min_bin:
                Smallest distogram bin (Angstroms)
            max_bin:
                Largest distogram bin (Angstroms)
            no_bins:
                Number of distogram bins
        """
        super(RecyclingEmbedder, self).__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bins = no_bins
        self.inf = inf

        self.linear = Linear(self.no_bins, self.c_z)
        self.layer_norm_m = LayerNorm(self.c_m)
        self.layer_norm_z = LayerNorm(self.c_z)

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            m:
                First row of the MSA embedding. [*, N_res, C_m]
            z:
                [*, N_res, N_res, C_z] pair embedding
            x:
                [*, N_res, 3] predicted C_beta coordinates
        Returns:
            m:
                [*, N_res, C_m] MSA embedding update
            z:
                [*, N_res, N_res, C_z] pair embedding update
        """
        bins = torch.linspace(
            self.min_bin,
            self.max_bin,
            self.no_bins,
            dtype=x.dtype,
            device=x.device,
            requires_grad=False,
        )

        # [*, N, C_m]
        m_update = self.layer_norm_m(m)

        # This squared method might become problematic in FP16 mode.
        # I'm using it because my homegrown method had a stubborn discrepancy I
        # couldn't find in time.
        squared_bins = bins ** 2
        upper = torch.cat(
            [squared_bins[1:], squared_bins.new_tensor([self.inf])], dim=-1
        )
        d = torch.sum(
            (x[..., None, :] - x[..., None, :, :]) ** 2, dim=-1, keepdims=True
        )
        # [*, N, N, no_bins]
        d = ((d > squared_bins) * (d < upper)).type(x.dtype)
        
        # [*, N, N, C_z]
        para_dim = d.shape[1]
        if CHUNK_SIZE == None:
            chunk_size = para_dim
        else:
            chunk_size = CHUNK_SIZE * 48
        
        for i in range(0, para_dim, chunk_size):
            di = self.linear(d[i:i + chunk_size, :, :])
            z[i:i + chunk_size, :, :] = di + self.layer_norm_z(z[i:i + chunk_size, :, :])

        return m_update, z


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

        self.scaling = self.c ** (-0.5)

        self.eps = 1e-10
        self.inf = 1e9

        self.to_q = Linear(qkv_dim, c * self.n_head, use_bias=False)
        self.to_kv = Linear(qkv_dim, 2 * c, initializer="linear", use_bias=False)

        self.gating_bias = nn.parameter.Parameter(data=torch.ones((n_head * c,)))
        self.gating_linear = Linear(
            qkv_dim, n_head * c, initializer="zero", use_bias=False
        )

        self.o_linear = Linear(n_head * c, out_dim, initializer="zero")

    def forward(self, m, mask):

        para_dim = m.shape[1]
        chunk_size = CHUNK_SIZE
        if CHUNK_SIZE == None:
            chunk_size = para_dim

        output = []
        for ax in range(0, para_dim, chunk_size):

            m_part = m[:, ax : ax + chunk_size, :, :]
            mask_part = mask[:, ax : ax + chunk_size, :]

            q = torch.sum(m_part * mask_part.unsqueeze(-1), dim=-2) / (
                torch.sum(mask_part, dim=-1)[..., None] + self.eps
            )

            q = self.to_q(q)
            q = q.view(q.shape[:-1] + (self.n_head, -1))

            k, v = self.to_kv(m_part).chunk(2, dim=-1)

            logits = torch.matmul(q, k.transpose(-1, -2))

            weights = mask_softmax(logits, mask_part)

            weighted_avg = torch.matmul(weights, v)
            weighted_avg = rearrange(weighted_avg, "b1 b2 h d -> b1 b2 (h d)")

            gate_values = self.gating_linear(m_part)
            weighted_avg = bias_sigmod_ele(
                gate_values, self.gating_bias, weighted_avg.unsqueeze(-2)
            )

            output.append(self.o_linear(weighted_avg))

        m = torch.cat(output, dim=1)

        return m