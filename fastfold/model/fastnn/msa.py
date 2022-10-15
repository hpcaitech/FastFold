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
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from fastfold.model.fastnn.kernel import LayerNorm, bias_dropout_add
from fastfold.model.fastnn.ops import (ChunkMSARowAttentionWithPairBias, ChunkTransition, 
                                       SelfAttention, GlobalAttention, Transition, 
                                       ChunkMSAColumnGlobalAttention, OutProductMean)
from fastfold.distributed import scatter, row_to_col
from fastfold.distributed.comm import gather, scatter, row_to_col, scatter
from fastfold.distributed.comm_async import gather_async, All_to_All_Async, All_to_All_Async_Opp
from fastfold.model.fastnn.triangle import PairCore


class MSARowAttentionWithPairBias(nn.Module):

    def __init__(self, d_node, d_pair, c=32, n_head=8, p_drop=0.15):
        super(MSARowAttentionWithPairBias, self).__init__()
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


class MSAColumnAttention(nn.Module):

    def __init__(self, d_node, c=32, n_head=8):
        super(MSAColumnAttention, self).__init__()
        self.d_node = d_node
        self.c = c
        self.n_head = n_head

        self.layernormM = LayerNorm(d_node)
        self.attention = SelfAttention(qkv_dim=d_node,
                                       c=c,
                                       n_head=n_head,
                                       out_dim=d_node,
                                       gating=True)

    def forward(self, M_raw, M_mask):
        M = M_raw.transpose(-2, -3)
        M = self.layernormM(M)

        M_mask = M_mask.transpose(-1, -2)
        # padding_bias = (1e9 * (M_mask - 1.))[:, :, None, None, :]

        M = self.attention(M, M_mask)

        M = M.transpose(-2, -3)
        return M_raw + M


class MSAColumnGlobalAttention(nn.Module):
    def __init__(self, d_node, c=8, n_head=8):
        super(MSAColumnGlobalAttention, self).__init__()

        self.d_node = d_node
        self.c = c
        self.n_head = n_head

        self.layernormM = LayerNorm(d_node)
        self.global_attention = GlobalAttention(
            qkv_dim=d_node, c=c, n_head=n_head, out_dim=d_node
        )

    def forward(self, M_raw, M_mask):
        M = M_raw.transpose(-2, -3)
        M = self.layernormM(M)

        M_mask = M_mask.transpose(-1, -2)

        M = self.global_attention(M, M_mask)

        M = M.transpose(-2, -3)
        return M_raw + M


class MSACore(nn.Module):

    def __init__(self, d_node, d_pair, p_drop=0.15):
        super(MSACore, self).__init__()

        self.MSARowAttentionWithPairBias = MSARowAttentionWithPairBias(d_node=d_node,
                                                                       d_pair=d_pair,
                                                                       p_drop=p_drop)

        self.MSAColumnAttention = MSAColumnAttention(d_node=d_node)
        self.MSATransition = Transition(d=d_node)

    def forward(self, node, pair, node_mask):
        # split node in row-axis
        node_mask_row = scatter(node_mask, dim=1)
        node = self.MSARowAttentionWithPairBias(node, pair, node_mask_row)

        node = row_to_col(node)
        node_mask_col = scatter(node_mask, dim=2)

        node = self.MSAColumnAttention(node, node_mask_col)
        node = self.MSATransition(node)

        return node


class ExtraMSACore(nn.Module):
    def __init__(self, d_node, d_pair, p_drop=0.15):
        super(ExtraMSACore, self).__init__()

        self.MSARowAttentionWithPairBias = ChunkMSARowAttentionWithPairBias(
            d_node=d_node, d_pair=d_pair, p_drop=p_drop, c=8
        )

        self.MSAColumnAttention = ChunkMSAColumnGlobalAttention(d_node=d_node, c=8)
        self.MSATransition = ChunkTransition(d=d_node)

    def forward(self, node, pair, node_mask):
        node_mask_row = scatter(node_mask, dim=1)
        node = self.MSARowAttentionWithPairBias(node, pair, node_mask_row)

        node = row_to_col(node)
        node_mask_col = scatter(node_mask, dim=2)

        node = self.MSAColumnAttention(node, node_mask_col)
        node = self.MSATransition(node)

        return node

    def inplace(self, node, pair, node_mask):
        node_mask_row = scatter(node_mask, dim=1)
        node = self.MSARowAttentionWithPairBias.inplace(node, pair[0], node_mask_row)

        node[0] = row_to_col(node[0])
        node_mask_col = scatter(node_mask, dim=2)

        node = self.MSAColumnAttention.inplace(node, node_mask_col)
        node = self.MSATransition.inplace(node)

        return node
    

class ExtraMSABlock(nn.Module):
    def __init__(
        self, c_m: int, c_z: int, first_block: bool, last_block: bool, is_multimer=False
    ):
        super(ExtraMSABlock, self).__init__()

        self.first_block = first_block
        self.last_block = last_block

        self.msa_stack = ExtraMSACore(c_m, c_z, p_drop=0.15)
        self.communication = OutProductMean(n_feat=c_m, n_feat_out=c_z, n_feat_proj=32)
        self.pair_stack = PairCore(d_pair=c_z)
        self.is_multimer = is_multimer

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        _mask_trans: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        dap_size = gpc.get_world_size(ParallelMode.TENSOR)

        seq_cnt = msa_mask.size(-2)
        seq_len = pair_mask.size(-1)
        seq_cnt_padding_size = (int(seq_cnt / dap_size) + 1) * dap_size - seq_cnt
        seq_len_padding_size = (int(seq_len / dap_size) + 1) * dap_size - seq_len

        if self.first_block:
            m = m.unsqueeze(0)
            z = z.unsqueeze(0)

            m = torch.nn.functional.pad(
                m, (0, 0, 0, seq_len_padding_size, 0, seq_cnt_padding_size)
            )
            z = torch.nn.functional.pad(
                z, (0, 0, 0, seq_len_padding_size, 0, seq_len_padding_size)
            )

            m = scatter(m, dim=1) if not self.is_multimer else scatter(m, dim=2)
            z = scatter(z, dim=1)

        msa_mask = msa_mask.unsqueeze(0)
        pair_mask = pair_mask.unsqueeze(0)

        msa_mask = torch.nn.functional.pad(
            msa_mask, (0, seq_len_padding_size, 0, seq_cnt_padding_size)
        )
        pair_mask = torch.nn.functional.pad(
            pair_mask, (0, seq_len_padding_size, 0, seq_len_padding_size)
        )

        if not self.is_multimer:
            m = self.msa_stack(m, z, msa_mask)
            z = self.communication(m, msa_mask, z)
            m, work = All_to_All_Async.apply(m, 1, 2)
            z = self.pair_stack(z, pair_mask)
            m = All_to_All_Async_Opp.apply(m, work, 1, 2)

        else:
            z = self.communication(m, msa_mask, z)
            z_ori = z
            m, work = All_to_All_Async.apply(m, 1, 2)
            z = self.pair_stack(z, pair_mask)
            m = All_to_All_Async_Opp.apply(m, work, 1, 2)
            m = self.msa_stack(m, z_ori, msa_mask)

        if self.last_block:

            m = gather(m, dim=1) if not self.is_multimer else gather(m, dim=2)
            z = gather(z, dim=1)

            m = m[:, :-seq_cnt_padding_size, :-seq_len_padding_size, :]
            z = z[:, :-seq_len_padding_size, :-seq_len_padding_size, :]

            m = m.squeeze(0)
            z = z.squeeze(0)

        return m, z

    def inplace(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        _mask_trans: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        dap_size = gpc.get_world_size(ParallelMode.TENSOR)

        seq_cnt = msa_mask.size(-2)
        seq_len = pair_mask.size(-1)
        seq_cnt_padding_size = (int(seq_cnt / dap_size) + 1) * dap_size - seq_cnt
        seq_len_padding_size = (int(seq_len / dap_size) + 1) * dap_size - seq_len

        if self.first_block:
            m[0] = m[0].unsqueeze(0)
            z[0] = z[0].unsqueeze(0)

            m[0] = torch.nn.functional.pad(
                m[0], (0, 0, 0, seq_len_padding_size, 0, seq_cnt_padding_size)
            )
            z[0] = torch.nn.functional.pad(
                z[0], (0, 0, 0, seq_len_padding_size, 0, seq_len_padding_size)
            )

            m[0] = scatter(m[0], dim=1) if not self.is_multimer else scatter(m[0], dim=2)
            z[0] = scatter(z[0], dim=1)

        msa_mask = msa_mask.unsqueeze(0)
        pair_mask = pair_mask.unsqueeze(0)

        msa_mask = torch.nn.functional.pad(
            msa_mask, (0, seq_len_padding_size, 0, seq_cnt_padding_size)
        )
        pair_mask = torch.nn.functional.pad(
            pair_mask, (0, seq_len_padding_size, 0, seq_len_padding_size)
        )

        if not self.is_multimer:
            m = self.msa_stack.inplace(m, z, msa_mask)
            z = self.communication.inplace(m[0], msa_mask, z)
            m[0], work = All_to_All_Async.apply(m[0], 1, 2)
            z = self.pair_stack.inplace(z, pair_mask)
            m[0] = All_to_All_Async_Opp.apply(m[0], work, 1, 2)
        else:
            # z = self.communication.inplace(m[0], msa_mask, z)
            # z_ori = [z[0].clone()]
            # m[0], work = All_to_All_Async.apply(m[0], 1, 2)
            # z = self.pair_stack.inplace(z, pair_mask)
            # m[0] = All_to_All_Async_Opp.apply(m[0], work, 1, 2)
            # m = self.msa_stack.inplace(m, z_ori, msa_mask)
            z = self.communication.inplace(m[0], msa_mask, z)
            m[0], work = All_to_All_Async.apply(m[0], 1, 2)
            m[0] = All_to_All_Async_Opp.apply(m[0], work, 1, 2)
            m = self.msa_stack.inplace(m, z, msa_mask)
            z = self.pair_stack.inplace(z, pair_mask)

        if self.last_block:

            m[0] = gather(m[0], dim=1) if not self.is_multimer else gather(m[0], dim=2)
            z[0] = gather(z[0], dim=1)

            m[0] = m[0][:, :-seq_cnt_padding_size, :-seq_len_padding_size, :]
            z[0] = z[0][:, :-seq_len_padding_size, :-seq_len_padding_size, :]

            m[0] = m[0].squeeze(0)
            z[0] = z[0].squeeze(0)

        return m, z


class ExtraMSAStack(nn.Module):
    """
    Implements Algorithm 18.
    """

    def __init__(self,
        c_m: int,
        c_z: int,
        no_blocks: int,
        clear_cache_between_blocks: bool = False,
        is_multimer: bool = False,
        **kwargs,
    ):
        super(ExtraMSAStack, self).__init__()
        
        self.clear_cache_between_blocks = clear_cache_between_blocks
        self.blocks = nn.ModuleList()
        for block_id in range(no_blocks):
            block = ExtraMSABlock(
                c_m=c_m,
                c_z=c_z,
                first_block=(block_id == 0),
                last_block=(block_id == no_blocks - 1),
                is_multimer=is_multimer,
            )
            self.blocks.append(block)

    def forward(self,
        m: torch.Tensor,
        z: torch.Tensor,
        chunk_size: int,
        msa_mask: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None,
        _mask_trans: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_extra, N_res, C_m] extra MSA embedding
            z:
                [*, N_res, N_res, C_z] pair embedding
            msa_mask:
                Optional [*, N_extra, N_res] MSA mask
            pair_mask:
                Optional [*, N_res, N_res] pair mask
        Returns:
            [*, N_res, N_res, C_z] pair update
        """ 
        #checkpoint_fn = get_checkpoint_fn()
        #blocks = [
        #    partial(b, msa_mask=msa_mask, pair_mask=pair_mask, chunk_size=chunk_size, _chunk_logits=None) for b in self.blocks
        #]

        #def dodo(b, *args):
        #    torch.cuda.empty_cache()
        #    return b(*args)

        #blocks = [partial(dodo, b) for b in blocks]

        #for b in blocks:
        #    if(torch.is_grad_enabled()):
        #        m, z = checkpoint_fn(b, *(m, z))
        #    else:
        #        m, z = b(m, z)

        for b in self.blocks:
            m, z = b(m, z, msa_mask, pair_mask, chunk_size=chunk_size)

            if(self.clear_cache_between_blocks):
                torch.cuda.empty_cache()

        return z

    def inplace(self,
        m: torch.Tensor,
        z: torch.Tensor,
        chunk_size: int,
        msa_mask: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None,
        _mask_trans: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_extra, N_res, C_m] extra MSA embedding
            z:
                [*, N_res, N_res, C_z] pair embedding
            msa_mask:
                Optional [*, N_extra, N_res] MSA mask
            pair_mask:
                Optional [*, N_res, N_res] pair mask
        Returns:
            [*, N_res, N_res, C_z] pair update
        """ 
        #checkpoint_fn = get_checkpoint_fn()
        #blocks = [
        #    partial(b, msa_mask=msa_mask, pair_mask=pair_mask, chunk_size=chunk_size, _chunk_logits=None) for b in self.blocks
        #]

        #def dodo(b, *args):
        #    torch.cuda.empty_cache()
        #    return b(*args)

        #blocks = [partial(dodo, b) for b in blocks]

        #for b in blocks:
        #    if(torch.is_grad_enabled()):
        #        m, z = checkpoint_fn(b, *(m, z))
        #    else:
        #        m, z = b(m, z)

        for b in self.blocks:
            m, z = b.inplace(m, z, msa_mask, pair_mask, chunk_size=chunk_size)

            if(self.clear_cache_between_blocks):
                torch.cuda.empty_cache()

        return z
