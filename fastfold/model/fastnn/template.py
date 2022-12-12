# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
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
from functools import partial
from typing import Optional, List

import torch
import torch.nn as nn

from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc

from fastfold.model.nn.primitives import Attention
from fastfold.utils.checkpointing import checkpoint_blocks
from fastfold.utils.tensor_utils import chunk_layer, permute_final_dims
from fastfold.model.fastnn.ops import (ChunkTransition, LayerNorm,
                                       ChunkTriangleAttentionStartingNode, ChunkTriangleAttentionEndingNode, 
                                       AsyncChunkTriangleMultiplicationOutgoing, AsyncChunkTriangleMultiplicationIncoming)
from fastfold.distributed.comm import gather, scatter, col_to_row, row_to_col, scatter


class TemplatePointwiseAttention(nn.Module):
    """
    Implements Algorithm 17.
    """
    def __init__(self, c_t, c_z, c_hidden, no_heads, inf, **kwargs):
        """
        Args:
            c_t:
                Template embedding channel dimension
            c_z:
                Pair embedding channel dimension
            c_hidden:
                Hidden channel dimension
        """
        super(TemplatePointwiseAttention, self).__init__()

        self.c_t = c_t
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf

        self.mha = Attention(
            self.c_z,
            self.c_t,
            self.c_t,
            self.c_hidden,
            self.no_heads,
            gating=False
        )

    def _chunk(self,
        z: torch.Tensor,
        t: torch.Tensor,
        biases: List[torch.Tensor],
        chunk_size: int,
    ) -> torch.Tensor:
        mha_inputs = {
            "q_x": z,
            "kv_x": t,
            "biases": biases,
        }
        return chunk_layer(
            self.mha,
            mha_inputs,
            chunk_size=chunk_size,
            no_batch_dims=len(z.shape[:-2]),
        )


    def forward(self, 
        t: torch.Tensor, 
        z: torch.Tensor, 
        template_mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Args:
            t:
                [*, N_templ, N_res, N_res, C_t] template embedding
            z:
                [*, N_res, N_res, C_t] pair embedding
            template_mask:
                [*, N_templ] template mask
        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        """
        if template_mask is None:
            template_mask = t.new_ones(t.shape[:-3])

        bias = self.inf * (template_mask[..., None, None, None, None, :] - 1)

        # [*, N_res, N_res, 1, C_z]
        z = z.unsqueeze(-2)

        # [*, N_res, N_res, N_temp, C_t]
        t = permute_final_dims(t, (1, 2, 0, 3))

        # [*, N_res, N_res, 1, C_z]
        biases = [bias]
        if chunk_size is not None:
            out = torch.empty_like(z)
            mask = torch.sum(template_mask.to(z.device)) > 0
            for t0 in range(t.shape[0]):
                for t1 in range(0, t.shape[1], chunk_size):
                    tt = t[t0, t1:t1 + chunk_size, :].unsqueeze(0)
                    tt = tt.to(z.device)
                    out[t0, t1:t1 + chunk_size, :] = self.mha(
                        q_x=z[t0, t1:t1 + chunk_size, :].unsqueeze(0),
                        kv_x=tt,
                        biases=biases
                    ).squeeze(0) * mask
        else:
            out = self.mha(q_x=z, kv_x=t, biases=biases)
            # [*, N_res, N_res, C_z]
            out = out * (torch.sum(template_mask) > 0)
            
        out = out.squeeze(-2)

        return out

    def inplace(self, 
        t: torch.Tensor, 
        z: torch.Tensor, 
        template_mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Args:
            t:
                [*, N_templ, N_res, N_res, C_t] template embedding
            z:
                [*, N_res, N_res, C_t] pair embedding
            template_mask:
                [*, N_templ] template mask
        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        """
        if template_mask is None:
            template_mask = t.new_ones(t.shape[:-3])

        bias = self.inf * (template_mask[..., None, None, None, None, :] - 1)

        # [*, N_res, N_res, 1, C_z]
        z = z.unsqueeze(-2)

        # [*, N_res, N_res, N_temp, C_t]
        t = permute_final_dims(t, (1, 2, 0, 3))

        # [*, N_res, N_res, 1, C_z]
        biases = [bias]
        if chunk_size is not None:
            mask = torch.sum(template_mask.to(z.device)) > 0
            for t0 in range(t.shape[0]):
                for t1 in range(0, t.shape[1], chunk_size):
                    tt = t[t0, t1:t1 + chunk_size, :].unsqueeze(0)
                    tt = tt.to(z.device)
                    z[t0, t1:t1 + chunk_size, :] += self.mha(
                        q_x=z[t0, t1:t1 + chunk_size, :].unsqueeze(0),
                        kv_x=tt,
                        biases=biases
                    ).squeeze(0) * mask
        else:
            t = self.mha(q_x=z, kv_x=t, biases=biases) * (torch.sum(template_mask) > 0)
            # [*, N_res, N_res, C_z]
            z += t
            
        z = z.squeeze(-2)

        return z


class TemplatePairBlock(nn.Module):
    def __init__(
        self,
        c_t: int,
        c_hidden_tri_att: int,
        c_hidden_tri_mul: int,
        no_heads: int,
        pair_transition_n: int,
        dropout_rate: float,
        inf: float,
        first_block: bool,
        last_block: bool,
        **kwargs,
    ):
        super(TemplatePairBlock, self).__init__()

        self.first_block = first_block
        self.last_block = last_block

        self.c_t = c_t
        self.c_hidden_tri_att = c_hidden_tri_att
        self.c_hidden_tri_mul = c_hidden_tri_mul
        self.n_head = no_heads
        self.p_drop = dropout_rate
        self.hidden_c = int(c_t / self.n_head)

        self.TriangleMultiplicationOutgoing = AsyncChunkTriangleMultiplicationOutgoing(
            self.c_t, p_drop=self.p_drop, c=self.c_hidden_tri_mul
        )
        self.TriangleMultiplicationIncoming = AsyncChunkTriangleMultiplicationIncoming(
            self.c_t, p_drop=self.p_drop, c=self.c_hidden_tri_mul
        )
        self.TriangleAttentionStartingNode = ChunkTriangleAttentionStartingNode(
            self.c_t, p_drop=self.p_drop, c=self.c_hidden_tri_att, n_head=self.n_head
        )
        self.TriangleAttentionEndingNode = ChunkTriangleAttentionEndingNode(
            self.c_t, p_drop=self.p_drop, c=self.c_hidden_tri_att, n_head=self.n_head
        )
        self.PairTransition = ChunkTransition(d=self.c_t, n=pair_transition_n)

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        _mask_trans: bool = True,
    ):

        dap_size = gpc.get_world_size(ParallelMode.TENSOR)

        seq_length = mask.size(-1)
        padding_size = (int(seq_length / dap_size) + 1) * dap_size - seq_length

        if self.first_block:
            z = torch.nn.functional.pad(z, (0, 0, 0, padding_size, 0, padding_size))
            z = scatter(z, dim=1)

        mask = torch.nn.functional.pad(mask, (0, padding_size, 0, padding_size))
        single_mask_row = scatter(mask, dim=1)
        single_mask_col = scatter(mask, dim=2)

        z = self.TriangleAttentionStartingNode(z, single_mask_row)
        z = row_to_col(z)
        z = self.TriangleAttentionEndingNode(z, single_mask_col)
        z = col_to_row(z)
        z = self.TriangleMultiplicationOutgoing(z, single_mask_row)
        z = row_to_col(z)
        z = self.TriangleMultiplicationIncoming(z, single_mask_col)
        z = self.PairTransition(z)
        z = col_to_row(z)

        # z = torch.cat(single_templates, dim=-4)
        if self.last_block:
            z = gather(z, dim=1)
            z = z[:, :-padding_size, :-padding_size, :]

        return z
    
    def inplace(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        _mask_trans: bool = True,
    ):
        dap_size = gpc.get_world_size(ParallelMode.TENSOR)
        seq_length = mask.size(-1)
        padding_size = (int(seq_length / dap_size) + 1) * dap_size - seq_length

        if self.first_block:
            z[0] = torch.nn.functional.pad(z[0], (0, 0, 0, padding_size, 0, padding_size))
            z[0] = scatter(z[0], dim=1)

        mask = torch.nn.functional.pad(mask, (0, padding_size, 0, padding_size))
        single_mask_row = scatter(mask, dim=1)
        single_mask_col = scatter(mask, dim=2)

        z = self.TriangleAttentionStartingNode.inplace(z, single_mask_row)
        z[0] = row_to_col(z[0])
        z = self.TriangleAttentionEndingNode.inplace(z, single_mask_col)
        z[0] = col_to_row(z[0])
        z[0] = self.TriangleMultiplicationOutgoing(z[0], single_mask_row)
        z[0] = row_to_col(z[0])
        z[0] = self.TriangleMultiplicationIncoming(z[0], single_mask_col)
        z = self.PairTransition.inplace(z)
        z[0] = col_to_row(z[0])

        # z = torch.cat(single_templates, dim=-4)
        if self.last_block:
            z[0] = gather(z[0], dim=1)
            z[0] = z[0][:, :-padding_size, :-padding_size, :]

        return z


class TemplatePairStack(nn.Module):
    """
    Implements Algorithm 16.
    """
    def __init__(
        self,
        c_t,
        c_hidden_tri_att,
        c_hidden_tri_mul,
        no_blocks,
        no_heads,
        pair_transition_n,
        dropout_rate,
        blocks_per_ckpt,
        inf=1e9,
        **kwargs,
    ):
        """
        Args:
            c_t:
                Template embedding channel dimension
            c_hidden_tri_att:
                Per-head hidden dimension for triangular attention
            c_hidden_tri_att:
                Hidden dimension for triangular multiplication
            no_blocks:
                Number of blocks in the stack
            pair_transition_n:
                Scale of pair transition (Alg. 15) hidden dimension
            dropout_rate:
                Dropout rate used throughout the stack
            blocks_per_ckpt:
                Number of blocks per activation checkpoint. None disables
                activation checkpointing
        """
        super(TemplatePairStack, self).__init__()

        self.blocks_per_ckpt = blocks_per_ckpt

        self.blocks = nn.ModuleList()
        for block_id in range(no_blocks):
            block = TemplatePairBlock(
                c_t=c_t,
                c_hidden_tri_att=c_hidden_tri_att,
                c_hidden_tri_mul=c_hidden_tri_mul,
                no_heads=no_heads,
                pair_transition_n=pair_transition_n,
                dropout_rate=dropout_rate,
                inf=inf,
                first_block=(block_id == 0),
                last_block=(block_id == no_blocks - 1),
            )
            self.blocks.append(block)

        self.layer_norm = LayerNorm(c_t)

    def forward(
        self,
        t: torch.tensor,
        mask: torch.tensor,
        chunk_size: int,
        _mask_trans: bool = True,
    ):
        """
        Args:
            t:
                [*, N_templ, N_res, N_res, C_t] template embedding
            mask:
                [*, N_templ, N_res, N_res] mask
        Returns:
            [*, N_templ, N_res, N_res, C_t] template embedding update
        """
        if(mask.shape[-3] == 1):
            expand_idx = list(mask.shape)
            expand_idx[-3] = t.shape[-4]
            mask = mask.expand(*expand_idx)

        t, = checkpoint_blocks(
            blocks=[
                partial(
                    b,
                    mask=mask,
                    chunk_size=chunk_size,
                    _mask_trans=_mask_trans,
                )
                for b in self.blocks
            ],
            args=(t,),
            blocks_per_ckpt=self.blocks_per_ckpt if self.training else None,
        )
        if not self.training:
            for i in range(0, t.shape[0]):
                t[i] = self.layer_norm(t[i])
        else:
            t = self.layer_norm(t)
        return t
    
    def inplace(
        self,
        t: torch.tensor,
        mask: torch.tensor,
        chunk_size: int,
        _mask_trans: bool = True,
    ):
        """
        Args:
            t:
                [*, N_templ, N_res, N_res, C_t] template embedding
            mask:
                [*, N_templ, N_res, N_res] mask
        Returns:
            [*, N_templ, N_res, N_res, C_t] template embedding update
        """
        if(mask.shape[-3] == 1):
            expand_idx = list(mask.shape)
            expand_idx[-3] = t[0].shape[-4]
            mask = mask.expand(*expand_idx)

        t, = checkpoint_blocks(
            blocks=[
                partial(
                    b.inplace,
                    mask=mask,
                    chunk_size=chunk_size,
                    _mask_trans=_mask_trans,
                )
                for b in self.blocks
            ],
            args=(t,),
            blocks_per_ckpt=self.blocks_per_ckpt if self.training else None,
        )

        for i in range(0, t[0].shape[0]):
            t[0][i] = self.layer_norm(t[0][i].to(mask.device)).to(t[0].device)
        return t
