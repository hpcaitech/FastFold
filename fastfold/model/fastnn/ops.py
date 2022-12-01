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
import math
from einops import rearrange
from typing import Tuple
from fastfold.model.fastnn.kernel import LayerNorm
from fastfold.model.fastnn.kernel import fused_softmax

from .initializer import glorot_uniform_af

from fastfold.model.fastnn.kernel import bias_sigmod_ele, bias_ele_dropout_residual, bias_dropout_add
from fastfold.distributed import gather, scatter
from fastfold.distributed.comm_async import gather_async, gather_async_opp, get_world_size, get_rank, broadcast_sync, broadcast_async, broadcast_async_opp


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


class ChunkTransition(nn.Module):

    def __init__(self, d, n=4):
        super(ChunkTransition, self).__init__()
        self.norm = LayerNorm(d)
        self.linear1 = Linear(d, n * d, initializer='relu')
        self.linear2 = Linear(n * d, d, initializer='zeros')

    def forward(self, src):
        if CHUNK_SIZE == None:
            out = self.norm(src)
            out = self.linear2(F.relu(self.linear1(out)))
        else:
            chunk_size = CHUNK_SIZE * 48
            para_dim = src.shape[1]
            out = torch.empty_like(src)
            for ax in range(0, para_dim, chunk_size):
                if DEBUG and ax > 10:
                    break
                x = self.norm(src[:, ax:ax + chunk_size, :, :])
                x = self.linear2(F.relu(self.linear1(x)))
                out[:, ax:ax + chunk_size, :, :] = x
        out.add_(src)
        return out

    def inplace(self, src):
        para_dim = src[0].shape[1]
        if CHUNK_SIZE == None:
            chunk_size = para_dim
        else:
            chunk_size = CHUNK_SIZE * 48

        for ax in range(0, para_dim, chunk_size):
            if DEBUG and ax > 10:
                break
            x = self.norm(src[0][:, ax:ax + chunk_size, :, :])
            x = self.linear2(F.relu(self.linear1(x)))
            src[0][:, ax:ax + chunk_size, :, :] += x
        return src


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
        self.n_feat_proj = n_feat_proj

    def forward(self, M, M_mask, Z_raw):
        Z = torch.empty_like(Z_raw)
        M = self.layernormM(M)
        right_act = self.linear_b(M)
        right_act_all, work = gather_async(right_act, dim=2)
        # right_act_all = gather(right_act, dim=2)

        left_act = self.linear_a(M)
        M_mask = M_mask.unsqueeze(-1)
        M_mask_col = scatter(M_mask, dim=2)
        left_act = M_mask_col * left_act
        norm = torch.einsum('bsid,bsjd->bijd', M_mask_col, M_mask) + 1e-3

        right_act_all = gather_async_opp(right_act_all, work, dim=2)
        right_act_all = M_mask * right_act_all

        if CHUNK_SIZE == None:
            out = torch.einsum('bsid, bsje->bijde', left_act, right_act_all)
            out = rearrange(out, 'b i j d e -> b i j (d e)')
            out = self.o_linear(out)
            Z = out / norm
        else:
            para_dim = left_act.shape[2]
            chunk_size = CHUNK_SIZE
            for ax in range(0, para_dim, chunk_size):
                left_act_part = left_act[:, :, ax:ax + chunk_size, :]
                O = torch.einsum('bsid,bsje->bijde', left_act_part, right_act_all)
                O = rearrange(O, 'b i j d e -> b i j (d e)')
                O = self.o_linear(O)
                norm0 = norm[:, ax:ax + chunk_size, :, :]
                Z[:, ax:ax + chunk_size, :, :] = O / norm0

        return Z + Z_raw

    def inplace(self, M, M_mask, Z_raw):
        
        chunk_size = CHUNK_SIZE
        if len(M.shape) == 4:
            para_dim = M.shape[1]
            left_act = torch.empty((M.shape[0], M.shape[1], M.shape[2], self.n_feat_proj), dtype=M.dtype, device=M.device)
            right_act = torch.empty((M.shape[0], M.shape[1], M.shape[2], self.n_feat_proj), dtype=M.dtype, device=M.device)
            if CHUNK_SIZE == None:
                chunk_size = para_dim
            else:
                chunk_size = chunk_size * 32
            for ax in range(0, para_dim, chunk_size):
                m = self.layernormM(M[:, ax:ax + chunk_size, :, :])
                right_act[:, ax:ax + chunk_size, :, :] = self.linear_b(m)
                left_act[:, ax:ax + chunk_size, :, :] = self.linear_a(m)
        else:
            para_dim = M.shape[0]
            left_act = torch.empty((M.shape[0], M.shape[1], self.n_feat_proj), dtype=M.dtype, device=M.device)
            right_act = torch.empty((M.shape[0], M.shape[1], self.n_feat_proj), dtype=M.dtype, device=M.device)
            if CHUNK_SIZE == None:
                chunk_size = para_dim
            else:
                chunk_size = chunk_size * 32
            for ax in range(0, para_dim, chunk_size):
                m = self.layernormM(M[ax:ax + chunk_size, :, :])
                right_act[ax:ax + chunk_size, :, :] = self.linear_b(m)
                left_act[ax:ax + chunk_size, :, :] = self.linear_a(m)

        right_act_all, work = gather_async(right_act, dim=2)
        # right_act_all = gather(right_act, dim=2)

        M_mask = M_mask.unsqueeze(-1)
        M_mask_col = scatter(M_mask, dim=2)
        left_act = M_mask_col * left_act
        norm = torch.einsum('bsid,bsjd->bijd', M_mask_col, M_mask) + 1e-3

        right_act_all = gather_async_opp(right_act_all, work, dim=2)
        right_act_all = M_mask * right_act_all

        para_dim = left_act.shape[2]
        chunk_size = CHUNK_SIZE
        if CHUNK_SIZE == None:
            chunk_size = para_dim

        for ax in range(0, para_dim, chunk_size):
            left_act_part = left_act[:, :, ax:ax + chunk_size, :]
            O = torch.einsum('bsid,bsje->bijde', left_act_part, right_act_all)
            O = rearrange(O, 'b i j d e -> b i j (d e)')
            O = self.o_linear(O)
            norm0 = norm[:, ax:ax + chunk_size, :, :]
            Z_raw[0][:, ax:ax + chunk_size, :, :] += O / norm0

        return Z_raw

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
        :param mask: None or [batch_size1, batch_size2, len_kv]
        :param nonbatched_bias: None or [batch_size1, n_head, len_q, len_kv]
        """

        if nonbatched_bias is not None:
            if nonbatched_bias[-1] == -1:
                bias = nonbatched_bias[0]
            else:
                # logits += nonbatched_bias.unsqueeze(1)
                bias = gather_async_opp(*nonbatched_bias, dim=1)
                bias = rearrange(bias, 'b q k h -> b h q k')
        
        if CHUNK_SIZE == None:
            qkv = self.to_qkv(in_data).chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b1 b2 n (h d) -> b1 b2 h n d', h=self.n_head), qkv)

            q = q * self.scaling

            logits = torch.matmul(q, k.transpose(-1, -2))

            if nonbatched_bias is not None:
                weights = fused_softmax(logits, mask, bias.unsqueeze(1))
            else:
                weights = fused_softmax(logits, mask)

            weighted_avg = torch.matmul(weights, v)
            weighted_avg = rearrange(weighted_avg, 'b1 b2 h n d -> b1 b2 n (h d)')

            if self.gating:
                gate_values = self.gating_linear(in_data)
                weighted_avg = bias_sigmod_ele(gate_values, self.gating_bias, weighted_avg)

            output = self.o_linear(weighted_avg)
            
        else:
            para_dim = in_data.shape[1]
            chunk_size = CHUNK_SIZE
            output = []
            for ax in range(0, para_dim, chunk_size):

                in_data_part = in_data[:, ax:ax + chunk_size, :, :]
                mask_part = mask[:, ax:ax + chunk_size, :]

                qkv = self.to_qkv(in_data_part).chunk(3, dim=-1)
                q, k, v = map(lambda t: rearrange(t, 'b1 b2 n (h d) -> b1 b2 h n d', h=self.n_head), qkv)

                q = q * self.scaling

                logits = torch.matmul(q, k.transpose(-1, -2))

                if nonbatched_bias is not None:
                    # logits += bias.unsqueeze(1)
                    # logits += (1e9 * (mask_part - 1))[..., :, None, None, :]
                    # weights = torch.nn.functional.softmax(logits, -1)
                    weights = fused_softmax(logits, mask_part, bias.unsqueeze(1))
                else:
                    # logits += (1e9 * (mask_part - 1))[..., :, None, None, :]
                    # weights = torch.nn.functional.softmax(logits, -1)
                    weights = fused_softmax(logits, mask_part)

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
            if DEBUG and i > 10:
                break
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
            if DEBUG and i > 10:
                break
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
            if DEBUG and i > 10:
                break
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

    def inplace(self, Z_raw, Z_mask):
        
        if CHUNK_SIZE == None:        
            Z = self.layernorm1(Z_raw[0])
            b = self.linear_b(Z)
            b, work = gather_async(b, dim=1)
            Z = self.attention(Z, Z_mask, (b, work))
            dropout_mask = torch.ones_like(Z[:, 0:1, :, :], device=Z.device, dtype=Z.dtype)
            Z_raw[0] = bias_dropout_add(Z,
                                    self.out_bias,
                                    dropout_mask,
                                    Z_raw[0],
                                    prob=self.p_drop,
                                    training=self.training)
            return Z_raw
        
        chunk_size = CHUNK_SIZE
        para_dim = Z_raw[0].shape[1]
        # z is big, but b is small. So we compute z in chunk to get b, and recompute z in chunk later instead of storing it
        b = torch.empty((Z_raw[0].shape[0], Z_raw[0].shape[1], Z_raw[0].shape[2], self.n_head), device=Z_raw[0].device, dtype=Z_raw[0].dtype)
        for i in range(0, para_dim, chunk_size):
            z = self.layernorm1(Z_raw[0][:, i:i + chunk_size, :, :])
            b[:, i:i + chunk_size, :, :] = self.linear_b(z)
        b, work = gather_async(b, dim=1)
        b = gather_async_opp(b, work, dim=1)
        b = rearrange(b, 'b q k h -> b h q k')
        
        # output = torch.empty_like(Z_raw)
        dropout_mask = torch.ones_like(z[:, 0:1, :, :], device=z.device, dtype=z.dtype)
        for i in range(0, para_dim, chunk_size):
            if DEBUG and i > 10:
                break
            z_raw = Z_raw[0][:, i:i + chunk_size, :, :]
            z = self.layernorm1(z_raw)
            z_mask = Z_mask[:, i:i + chunk_size, :]
            
            z = self.attention(z, z_mask, (b, -1))
            z =  bias_dropout_add(z,
                                    self.out_bias,
                                    dropout_mask,
                                    z_raw,
                                    prob=self.p_drop,
                                    training=self.training)
            Z_raw[0][:, i:i + chunk_size, :, :] = z
        
        return Z_raw


class ChunkMSARowAttentionWithPairBias(nn.Module):

    def __init__(self, d_node, d_pair, c=32, n_head=8, p_drop=0.15):
        super(ChunkMSARowAttentionWithPairBias, self).__init__()
        self.d_node = d_node
        self.d_pair = d_pair
        self.c = c
        self.n_head = n_head
        self.p_drop = p_drop

        self.layernormM = LayerNorm(d_node)
        self.layernormZ = LayerNorm(d_pair)

        _init_weights = torch.nn.init.normal_(torch.zeros([n_head, d_pair]),
                                              std=1.0 / math.sqrt(d_pair))
        self.linear_b_weights = nn.parameter.Parameter(data=_init_weights, requires_grad=True)

        self.attention = SelfAttention(qkv_dim=d_node,
                                       c=c,
                                       n_head=n_head,
                                       out_dim=d_node,
                                       gating=True,
                                       last_bias_fuse=True)

        self.out_bias = nn.parameter.Parameter(data=torch.zeros((d_node,)), requires_grad=True)

    def forward(self, M_raw, Z, M_mask):
        
        if CHUNK_SIZE == None:
            ## Input projections
            M = self.layernormM(M_raw)
            Z = self.layernormZ(Z)
            b = F.linear(Z, self.linear_b_weights)
            b, work = gather_async(b, dim=1)
            # b = rearrange(b, 'b q k h -> b h q k')
            # padding_bias = (1e9 * (M_mask - 1.))[:, :, None, None, :]
            M = self.attention(M, M_mask, (b, work))
            dropout_mask = torch.ones_like(M[:, 0:1, :, :], device=M.device, dtype=M.dtype)
            return bias_dropout_add(M, self.out_bias, dropout_mask, M_raw, prob=self.p_drop, training=self.training)

        chunk_size = CHUNK_SIZE
        para_dim_z = Z.shape[1]
        para_dim_m = M_raw.shape[1]
        # z is big, but b is small. So we compute z in chunk to get b, and recompute z in chunk later instead of storing it
        b = torch.empty((Z.shape[0], Z.shape[1], Z.shape[2], self.n_head), device=Z.device, dtype=Z.dtype)
        for i in range(0, para_dim_z, chunk_size):
            z = self.layernormZ(Z[:, i:i + chunk_size, :, :])
            b[:, i:i + chunk_size, :, :] = F.linear(z, self.linear_b_weights)
        b, work = gather_async(b, dim=1)
        b = gather_async_opp(b, work, dim=1)
        b = rearrange(b, 'b q k h -> b h q k')
        
        output = torch.empty_like(M_raw)
        dropout_mask = torch.ones_like(M_raw[:, 0:1, :, :], device=M_raw.device, dtype=M_raw.dtype)
        for i in range(0, para_dim_m, chunk_size):
            if DEBUG and i > 10:
                break
            m_raw = M_raw[:, i:i + chunk_size, :, :]
            m = self.layernormM(m_raw)
            m_mask = M_mask[:, i:i + chunk_size, :]
            
            m = self.attention(m, m_mask, (b, -1))
            m =  bias_dropout_add(m,
                                    self.out_bias,
                                    dropout_mask,
                                    m_raw,
                                    prob=self.p_drop,
                                    training=self.training)
            output[:, i:i + chunk_size, :, :] = m

        return output
    
    def inplace(self, M_raw, Z, M_mask):
        
        if CHUNK_SIZE == None:
            ## Input projections
            M = self.layernormM(M_raw[0])
            Z = self.layernormZ(Z)
            b = F.linear(Z, self.linear_b_weights)
            b, work = gather_async(b, dim=1)
            # b = rearrange(b, 'b q k h -> b h q k')
            # padding_bias = (1e9 * (M_mask - 1.))[:, :, None, None, :]
            M = self.attention(M, M_mask, (b, work))
            dropout_mask = torch.ones_like(M[:, 0:1, :, :], device=M.device, dtype=M.dtype)
            M_raw[0] = bias_dropout_add(M, self.out_bias, dropout_mask, M_raw[0], prob=self.p_drop, training=self.training)
            return M_raw

        chunk_size = CHUNK_SIZE
        para_dim_z = Z.shape[1]
        para_dim_m = M_raw[0].shape[1]
        # z is big, but b is small. So we compute z in chunk to get b, and recompute z in chunk later instead of storing it
        b = torch.empty((Z.shape[0], Z.shape[1], Z.shape[2], self.n_head), device=Z.device, dtype=Z.dtype)
        for i in range(0, para_dim_z, chunk_size):
            z = self.layernormZ(Z[:, i:i + chunk_size, :, :])
            b[:, i:i + chunk_size, :, :] = F.linear(z, self.linear_b_weights)
        b, work = gather_async(b, dim=1)
        b = gather_async_opp(b, work, dim=1)
        b = rearrange(b, 'b q k h -> b h q k')
        
        dropout_mask = torch.ones_like(M_raw[0][:, 0:1, :, :], device=M_raw[0].device, dtype=M_raw[0].dtype)
        for i in range(0, para_dim_m, chunk_size):
            if DEBUG and i > 10:
                break
            m_raw = M_raw[0][:, i:i + chunk_size, :, :]
            m = self.layernormM(m_raw)
            m_mask = M_mask[:, i:i + chunk_size, :]
            
            m = self.attention(m, m_mask, (b, -1))
            m =  bias_dropout_add(m,
                                    self.out_bias,
                                    dropout_mask,
                                    m_raw,
                                    prob=self.p_drop,
                                    training=self.training)
            M_raw[0][:, i:i + chunk_size, :, :] = m

        return M_raw

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
            if DEBUG and i > 10:
                break
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

    
    def inplace(self, Z_raw, Z_mask):
        
        if CHUNK_SIZE == None:  
            Z = Z_raw[0].transpose(-2, -3)
            Z_mask = Z_mask.transpose(-1, -2)

            Z = self.layernorm1(Z)
            b = self.linear_b(Z)
            b, work = gather_async(b, dim=1)
            Z = self.attention(Z, Z_mask, (b, work))
            Z = Z.transpose(-2, -3)
            dropout_mask = torch.ones_like(Z[:, :, 0:1, :], device=Z.device, dtype=Z.dtype)
            Z_raw[0] =  bias_dropout_add(Z,
                                    self.out_bias,
                                    dropout_mask,
                                    Z_raw[0],
                                    prob=self.p_drop,
                                    training=self.training)
            return Z_raw

        para_dim = Z_raw[0].shape[2]
        chunk_size = CHUNK_SIZE
        # z is big, but b is small. So we compute z in chunk to get b, and recompute z in chunk later instead of storing it
        b = torch.empty((Z_raw[0].shape[0], Z_raw[0].shape[2], Z_raw[0].shape[1], self.n_head), device=Z_raw[0].device, dtype=Z_raw[0].dtype)
        for i in range(0, para_dim, chunk_size):
            z = Z_raw[0][:, :, i:i + chunk_size, :].transpose(-2, -3)
            z = self.layernorm1(z)
            b[:, i:i + chunk_size, :, :] = self.linear_b(z)
        b, work = gather_async(b, dim=1)
        b = gather_async_opp(b, work, dim=1)
        b = rearrange(b, 'b q k h -> b h q k')
        
        dropout_mask = torch.ones_like(Z_raw[0][:, :, 0:1, :], device=z.device, dtype=z.dtype)
        for i in range(0, para_dim, chunk_size):
            if DEBUG and i > 10:
                break
            z_raw = Z_raw[0][:, :, i:i + chunk_size, :]
            z = self.layernorm1(z_raw.transpose(-2, -3))
            z_mask = Z_mask[:, :, i:i + chunk_size].transpose(-1, -2)

            z = self.attention(z, z_mask, (b, -1)).transpose(-2, -3)
            z =  bias_dropout_add(z,
                                    self.out_bias,
                                    dropout_mask,
                                    z_raw,
                                    prob=self.p_drop,
                                    training=self.training)
            Z_raw[0][:, :, i:i + chunk_size, :] = z
        
        return Z_raw


class ChunkMSAColumnGlobalAttention(nn.Module):
    def __init__(self, d_node, c=8, n_head=8):
        super(ChunkMSAColumnGlobalAttention, self).__init__()

        self.d_node = d_node
        self.c = c
        self.n_head = n_head

        self.layernormM = LayerNorm(d_node)
        self.global_attention = GlobalAttention(
            qkv_dim=d_node, c=c, n_head=n_head, out_dim=d_node
        )

    def forward(self, M_raw, M_mask):
        if CHUNK_SIZE is None:
            m = self.layernormM(M_raw.transpose(-2, -3))
            m = self.global_attention(m, M_mask.transpose(-1, -2))
            m = m.transpose(-2, -3)
            M_raw = M_raw + m

        else:
            chunk_size = CHUNK_SIZE
            para_dim = M_raw.shape[2]
            for i in range(0, para_dim, chunk_size):
                m = M_raw[:, :, i:i + chunk_size, :].transpose(-2, -3)
                m = self.layernormM(m)
                m_mask = M_mask[:, :, i:i + chunk_size].transpose(-1, -2)
                m = self.global_attention(m, m_mask)
                m = m.transpose(-2, -3)
                M_raw[:, :, i:i + chunk_size, :] += m
        
        return M_raw

    def inplace(self, M_raw, M_mask):
        
        para_dim = M_raw[0].shape[2]
        if CHUNK_SIZE is None:
            chunk_size = para_dim
        else:
            chunk_size = CHUNK_SIZE
        
        for i in range(0, para_dim, chunk_size):
            if DEBUG and i > 10:
                break
            m = M_raw[0][:, :, i:i + chunk_size, :].transpose(-2, -3)
            m = self.layernormM(m)
            m_mask = M_mask[:, :, i:i + chunk_size].transpose(-1, -2)
            m = self.global_attention(m, m_mask)
            m = m.transpose(-2, -3)
            M_raw[0][:, :, i:i + chunk_size, :] += m
        
        return M_raw


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
        
        if CHUNK_SIZE == None:
            d = self.linear(d)
            z = d + self.layer_norm_z(z)
        else:
            chunk_size = CHUNK_SIZE * 48
            para_dim = d.shape[1]
        
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

        if CHUNK_SIZE == None:
            q = torch.sum(m * mask.unsqueeze(-1), dim=-2) / (
                torch.sum(mask, dim=-1)[..., None] + self.eps
            )
            q = q * self.scaling
            q = self.to_q(q)
            q = q.view(q.shape[:-1] + (self.n_head, -1))

            k, v = self.to_kv(m).chunk(2, dim=-1)

            logits = torch.matmul(q, k.transpose(-1, -2))

            weights = fused_softmax(logits, mask)

            weighted_avg = torch.matmul(weights, v)
            weighted_avg = rearrange(weighted_avg, "b1 b2 h d -> b1 b2 (h d)")

            gate_values = self.gating_linear(m)
            weighted_avg = bias_sigmod_ele(
                gate_values, self.gating_bias, weighted_avg.unsqueeze(-2)
            )

            m = self.o_linear(weighted_avg)

        else:
            para_dim = m.shape[1]
            chunk_size = CHUNK_SIZE

            output = []
            for ax in range(0, para_dim, chunk_size):

                m_part = m[:, ax : ax + chunk_size, :, :]
                mask_part = mask[:, ax : ax + chunk_size, :]

                q = torch.sum(m_part * mask_part.unsqueeze(-1), dim=-2) / (
                    torch.sum(mask_part, dim=-1)[..., None] + self.eps
                )
                q = q * self.scaling
                q = self.to_q(q)
                q = q.view(q.shape[:-1] + (self.n_head, -1))

                k, v = self.to_kv(m_part).chunk(2, dim=-1)

                logits = torch.matmul(q, k.transpose(-1, -2))

                weights = fused_softmax(logits, mask_part)

                weighted_avg = torch.matmul(weights, v)
                weighted_avg = rearrange(weighted_avg, "b1 b2 h d -> b1 b2 (h d)")

                gate_values = self.gating_linear(m_part)
                weighted_avg = bias_sigmod_ele(
                    gate_values, self.gating_bias, weighted_avg.unsqueeze(-2)
                )

                output.append(self.o_linear(weighted_avg))

            m = torch.cat(output, dim=1)

        return m

class InputEmbedder(nn.Module):
    """
    Embeds a subset of the input features.

    Implements Algorithms 3 (InputEmbedder) and 4 (relpos).
    """

    def __init__(
        self,
        tf_dim: int,
        msa_dim: int,
        c_z: int,
        c_m: int,
        relpos_k: int,
        **kwargs,
    ):
        """
        Args:
            tf_dim:
                Final dimension of the target features
            msa_dim:
                Final dimension of the MSA features
            c_z:
                Pair embedding dimension
            c_m:
                MSA embedding dimension
            relpos_k:
                Window size used in relative positional encoding
        """
        super(InputEmbedder, self).__init__()

        self.tf_dim = tf_dim
        self.msa_dim = msa_dim

        self.c_z = c_z
        self.c_m = c_m

        self.linear_tf_z_i = Linear(tf_dim, c_z)
        self.linear_tf_z_j = Linear(tf_dim, c_z)
        self.linear_tf_m = Linear(tf_dim, c_m)
        self.linear_msa_m = Linear(msa_dim, c_m)

        # RPE stuff
        self.relpos_k = relpos_k
        self.no_bins = 2 * relpos_k + 1
        self.linear_relpos = Linear(self.no_bins, c_z)

    def forward(
        self,
        tf: torch.Tensor,
        ri: torch.Tensor,
        msa: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tf:
                "target_feat" features of shape [*, N_res, tf_dim]
            ri:
                "residue_index" features of shape [*, N_res]
            msa:
                "msa_feat" features of shape [*, N_clust, N_res, msa_dim]
        Returns:
            msa_emb:
                [*, N_clust, N_res, C_m] MSA embedding
            pair_emb:
                [*, N_res, N_res, C_z] pair embedding

        """
        # [*, N_res, c_z]
        tf_emb_i = self.linear_tf_z_i(tf)
        tf_emb_j = self.linear_tf_z_j(tf)

        # [*, N_res, N_res, c_z]        
        ri = ri.type(tf_emb_i.dtype)
        d = ri[..., None] - ri[..., None, :]
        boundaries = torch.arange(
            start=-self.relpos_k, end=self.relpos_k + 1, device=d.device
        )

        reshaped_bins = boundaries.view(((1,) * len(d.shape)) + (len(boundaries),))
        pair_emb = d[..., None] - reshaped_bins
        pair_emb = torch.argmin(torch.abs(pair_emb), dim=-1)
        pair_emb = nn.functional.one_hot(pair_emb, num_classes=len(boundaries)).float().type(ri.dtype)
        pair_emb = self.linear_relpos(pair_emb)
        pair_emb += tf_emb_i[..., None, :]
        pair_emb += tf_emb_j[..., None, :, :]

        # [*, N_clust, N_res, c_m]
        n_clust = msa.shape[-3]
        tf_m = (
            self.linear_tf_m(tf)
            .unsqueeze(-3)
            .expand(((-1,) * len(tf.shape[:-2]) + (n_clust, -1, -1)))
        )
        msa_emb = self.linear_msa_m(msa) + tf_m

        return msa_emb, pair_emb
