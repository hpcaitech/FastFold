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
from typing import Optional, Tuple

import torch
import torch.nn as nn

from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc

from fastfold.model.fastnn import MSAStack, OutProductMean, PairStack, ExtraMSAStack
from fastfold.model.fastnn.ops import Transition
from fastfold.model.fastnn.triangle import TriangleAttentionEndingNode, TriangleAttentionStartingNode, \
                                           TriangleMultiplicationIncoming, TriangleMultiplicationOutgoing
from fastfold.distributed.comm import gather, scatter
from fastfold.distributed.comm import col_to_row, row_to_col, scatter
from fastfold.distributed.comm_async import All_to_All_Async, All_to_All_Async_Opp



class EvoformerBlock(nn.Module):

    def __init__(self, c_m: int, c_z: int, first_block: bool, last_block: bool):
        super(EvoformerBlock, self).__init__()

        self.first_block = first_block
        self.last_block = last_block

        self.msa_stack = MSAStack(c_m, c_z, p_drop=0.15)
        self.communication = OutProductMean(n_feat=c_m, n_feat_out=c_z, n_feat_proj=32)
        self.pair_stack = PairStack(d_pair=c_z)

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

        seq_length = pair_mask.size(-1)
        padding_size = (int(seq_length / dap_size) + 1) * dap_size - seq_length

        if self.first_block:
            m = m.unsqueeze(0)
            z = z.unsqueeze(0)

            m = torch.nn.functional.pad(m, (0, 0, 0, padding_size))
            z = torch.nn.functional.pad(z, (0, 0, 0, padding_size, 0, padding_size))

            m = scatter(m, dim=1)
            z = scatter(z, dim=1)

        msa_mask = msa_mask.unsqueeze(0)
        pair_mask = pair_mask.unsqueeze(0)

        msa_mask = torch.nn.functional.pad(msa_mask, (0, padding_size))
        pair_mask = torch.nn.functional.pad(pair_mask, (0, padding_size, 0, padding_size))

        m = self.msa_stack(m, z, msa_mask)

        z = z + self.communication(m, msa_mask)
        m, work = All_to_All_Async.apply(m, 1, 2)
        z = self.pair_stack(z, pair_mask)
        m = All_to_All_Async_Opp.apply(m, work, 1, 2)

        if self.last_block:
            m = m.squeeze(0)
            z = z.squeeze(0)

            m = gather(m, dim=0)
            z = gather(z, dim=0)

            m = m[:, :-padding_size, :]
            z = z[:-padding_size, :-padding_size, :]

        return m, z


class ExtraMSABlock(nn.Module):
    def __init__(
        self, c_m: int, c_z: int, first_block: bool, last_block: bool, is_multimer=False
    ):
        super(ExtraMSABlock, self).__init__()

        self.first_block = first_block
        self.last_block = last_block

        self.msa_stack = ExtraMSAStack(c_m, c_z, p_drop=0.15)
        self.communication = OutProductMean(n_feat=c_m, n_feat_out=c_z, n_feat_proj=32)
        self.pair_stack = PairStack(d_pair=c_z)
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

            z = z + self.communication(m, msa_mask)
            m, work = All_to_All_Async.apply(m, 1, 2)
            z = self.pair_stack(z, pair_mask)
            m = All_to_All_Async_Opp.apply(m, work, 1, 2)

        else:
            z = z + self.communication(m, msa_mask)
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


class TemplatePairStackBlock(nn.Module):
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
        super(TemplatePairStackBlock, self).__init__()

        self.first_block = first_block
        self.last_block = last_block

        self.c_t = c_t
        self.c_hidden_tri_att = c_hidden_tri_att
        self.c_hidden_tri_mul = c_hidden_tri_mul
        self.n_head = no_heads
        self.p_drop = dropout_rate
        self.hidden_c = int(c_t / self.n_head)

        self.TriangleMultiplicationOutgoing = TriangleMultiplicationOutgoing(
            self.c_t, p_drop=self.p_drop, c=self.c_hidden_tri_mul
        )
        self.TriangleMultiplicationIncoming = TriangleMultiplicationIncoming(
            self.c_t, p_drop=self.p_drop, c=self.c_hidden_tri_mul
        )
        self.TriangleAttentionStartingNode = TriangleAttentionStartingNode(
            self.c_t, p_drop=self.p_drop, c=self.c_hidden_tri_att, n_head=self.n_head
        )
        self.TriangleAttentionEndingNode = TriangleAttentionEndingNode(
            self.c_t, p_drop=self.p_drop, c=self.c_hidden_tri_att, n_head=self.n_head
        )
        self.PairTransition = Transition(d=self.c_t, n=pair_transition_n)

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

        single_templates = [t.unsqueeze(-4) for t in torch.unbind(z, dim=-4)]
        single_templates_masks = [m.unsqueeze(-3) for m in torch.unbind(mask, dim=-3)]

        for i in range(len(single_templates)):
            single = single_templates[i]
            single_mask = single_templates_masks[i]

            single_mask_row = scatter(single_mask, dim=1)
            single_mask_col = scatter(single_mask, dim=2)

            single = self.TriangleMultiplicationOutgoing(single, single_mask_row)
            single = row_to_col(single)
            single = self.TriangleMultiplicationIncoming(single, single_mask_col)
            single = col_to_row(single)
            single = self.TriangleAttentionStartingNode(single, single_mask_row)
            single = row_to_col(single)
            single = self.TriangleAttentionEndingNode(single, single_mask_col)
            single = self.PairTransition(single)
            single = col_to_row(single)

            single_templates[i] = single

        z = torch.cat(single_templates, dim=-4)

        if self.last_block:
            z = gather(z, dim=1)
            z = z[:, :-padding_size, :-padding_size, :]

        return z