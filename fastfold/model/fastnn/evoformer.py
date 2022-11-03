from typing import Optional, Tuple
from functools import partial

import torch
import torch.nn as nn

from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc

from fastfold.model.fastnn import MSACore, OutProductMean, PairCore
from fastfold.model.fastnn.ops import Linear
from fastfold.distributed.comm import gather, scatter, col_to_row
from fastfold.distributed.comm_async import All_to_All_Async, All_to_All_Async_Opp
from fastfold.utils.checkpointing import checkpoint_blocks


class Evoformer(nn.Module):

    def __init__(self, c_m: int, c_z: int, first_block: bool, last_block: bool, is_multimer: bool=False):
        super(Evoformer, self).__init__()

        self.first_block = first_block
        self.last_block = last_block

        self.msa = MSACore(c_m, c_z, p_drop=0.15)
        self.communication = OutProductMean(n_feat=c_m, n_feat_out=c_z, n_feat_proj=32)
        self.pair = PairCore(d_pair=c_z)
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

        seq_length = pair_mask.size(-1)
        padding_size = (int(seq_length / dap_size) + 1) * dap_size - seq_length

        if self.first_block:
            m = m.unsqueeze(0)
            z = z.unsqueeze(0)

            m = torch.nn.functional.pad(m, (0, 0, 0, padding_size))
            z = torch.nn.functional.pad(z, (0, 0, 0, padding_size, 0, padding_size))

            if self.is_multimer:
                m = scatter(m, dim=2)
            else:
                m = scatter(m, dim=1)
            z = scatter(z, dim=1)

        msa_mask = msa_mask.unsqueeze(0)
        pair_mask = pair_mask.unsqueeze(0)

        msa_mask = torch.nn.functional.pad(msa_mask, (0, padding_size))
        pair_mask = torch.nn.functional.pad(pair_mask, (0, padding_size, 0, padding_size))

        if not self.is_multimer:
            m = self.msa(m, z, msa_mask)
            z = self.communication(m, msa_mask, z)
            m, work = All_to_All_Async.apply(m, 1, 2)
            z = self.pair(z, pair_mask)
            m = All_to_All_Async_Opp.apply(m, work, 1, 2)
        else:
            z = self.communication(m, msa_mask, z)
            z_ori = z
            m, work = All_to_All_Async.apply(m, 1, 2)
            z = self.pair(z, pair_mask)
            m = All_to_All_Async_Opp.apply(m, work, 1, 2)
            m = self.msa(m, z_ori, msa_mask)

        if self.last_block:
            m = m.squeeze(0)
            z = z.squeeze(0)

            if self.is_multimer:
                m = gather(m, dim=1)
            else:
                m = gather(m, dim=0)
            z = gather(z, dim=0)

            m = m[:, :-padding_size, :]
            z = z[:-padding_size, :-padding_size, :]

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

        seq_length = pair_mask.size(-1)
        padding_size = (int(seq_length / dap_size) + 1) * dap_size - seq_length

        if self.first_block:
            m[0] = m[0].unsqueeze(0)
            z[0] = z[0].unsqueeze(0)

            m[0] = torch.nn.functional.pad(m[0], (0, 0, 0, padding_size))
            z[0] = torch.nn.functional.pad(z[0], (0, 0, 0, padding_size, 0, padding_size))

            if self.is_multimer:
                m[0] = scatter(m[0], dim=2)
            else:
                m[0] = scatter(m[0], dim=1)
            z[0] = scatter(z[0], dim=1)

        msa_mask = msa_mask.unsqueeze(0)
        pair_mask = pair_mask.unsqueeze(0)

        msa_mask = torch.nn.functional.pad(msa_mask, (0, padding_size))
        pair_mask = torch.nn.functional.pad(pair_mask, (0, padding_size, 0, padding_size))

        if not self.is_multimer:
            m[0] = self.msa(m[0], z[0], msa_mask)
            z = self.communication.inplace(m[0], msa_mask, z)
            m[0], work = All_to_All_Async.apply(m[0], 1, 2)
            z = self.pair.inplace(z, pair_mask)
            m[0] = All_to_All_Async_Opp.apply(m[0], work, 1, 2)
        else:
            z = self.communication.inplace(m[0], msa_mask, z)
            m[0] = col_to_row(m[0])
            m[0] = self.msa(m[0], z[0], msa_mask)
            z = self.pair.inplace(z, pair_mask)

        if self.last_block:
            m[0] = m[0].squeeze(0)
            z[0] = z[0].squeeze(0)

            if self.is_multimer:
                m[0] = gather(m[0], dim=1)
            else:
                m[0] = gather(m[0], dim=0)
            z[0] = gather(z[0], dim=0)

            m[0] = m[0][:, :-padding_size, :]
            z[0] = z[0][:-padding_size, :-padding_size, :]

        return m, z


class EvoformerStack(nn.Module):
    """
    Main Evoformer trunk.
    Implements Algorithm 6.
    """

    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_s: int,
        no_blocks: int,
        blocks_per_ckpt: int,
        clear_cache_between_blocks: bool = False, 
        is_multimer: bool = False,
        **kwargs,
    ):
        """
        Args:
            c_m:
                MSA channel dimension
            c_z:
                Pair channel dimension
            c_hidden_msa_att:
                Hidden dimension in MSA attention
            c_hidden_opm:
                Hidden dimension in outer product mean module
            c_hidden_mul:
                Hidden dimension in multiplicative updates
            c_hidden_pair_att:
                Hidden dimension in triangular attention
            c_s:
                Channel dimension of the output "single" embedding
            no_heads_msa:
                Number of heads used for MSA attention
            no_heads_pair:
                Number of heads used for pair attention
            no_blocks:
                Number of Evoformer blocks in the stack
            transition_n:
                Factor by which to multiply c_m to obtain the MSATransition
                hidden dimension
            msa_dropout:
                Dropout rate for MSA activations
            pair_dropout:
                Dropout used for pair activations
            blocks_per_ckpt:
                Number of Evoformer blocks in each activation checkpoint
            clear_cache_between_blocks:
                Whether to clear CUDA's GPU memory cache between blocks of the
                stack. Slows down each block but can reduce fragmentation
        """
        super(EvoformerStack, self).__init__()

        self.blocks_per_ckpt = blocks_per_ckpt
        self.clear_cache_between_blocks = clear_cache_between_blocks

        self.blocks = nn.ModuleList()

        for block_id in range(no_blocks):
            block = Evoformer(
                c_m=c_m,
                c_z=c_z,
                first_block=(block_id == 0),
                last_block=(block_id == no_blocks - 1),
                is_multimer=is_multimer,
            )
            self.blocks.append(block)

        self.linear = Linear(c_m, c_s)

    def forward(self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int,
        _mask_trans: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            z:
                [*, N_res, N_res, C_z] pair embedding
            msa_mask:
                [*, N_seq, N_res] MSA mask
            pair_mask:
                [*, N_res, N_res] pair mask
        Returns:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            z:
                [*, N_res, N_res, C_z] pair embedding
            s:
                [*, N_res, C_s] single embedding (or None if extra MSA stack)
        """
        blocks = [
            partial(
                b,
                msa_mask=msa_mask,
                pair_mask=pair_mask,
                chunk_size=chunk_size,
                _mask_trans=_mask_trans,
            )
            for b in self.blocks
        ]

        if(self.clear_cache_between_blocks):
            def block_with_cache_clear(block, *args):
                torch.cuda.empty_cache()
                return block(*args)

            blocks = [partial(block_with_cache_clear, b) for b in blocks]

        m, z = checkpoint_blocks(
            blocks,
            args=(m, z),
            blocks_per_ckpt=self.blocks_per_ckpt if self.training else None,
        )

        s = self.linear(m[..., 0, :, :])
        
        return m, z, s

    def inplace(self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int,
        _mask_trans: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            z:
                [*, N_res, N_res, C_z] pair embedding
            msa_mask:
                [*, N_seq, N_res] MSA mask
            pair_mask:
                [*, N_res, N_res] pair mask
        Returns:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            z:
                [*, N_res, N_res, C_z] pair embedding
            s:
                [*, N_res, C_s] single embedding (or None if extra MSA stack)
        """
        blocks = [
            partial(
                b.inplace,
                msa_mask=msa_mask,
                pair_mask=pair_mask,
                chunk_size=chunk_size,
                _mask_trans=_mask_trans,
            )
            for b in self.blocks
        ]

        if(self.clear_cache_between_blocks):
            def block_with_cache_clear(block, *args):
                torch.cuda.empty_cache()
                return block(*args)

            blocks = [partial(block_with_cache_clear, b) for b in blocks]

        m, z = checkpoint_blocks(
            blocks,
            args=(m, z),
            blocks_per_ckpt=self.blocks_per_ckpt if self.training else None,
        )

        s = self.linear(m[0][..., 0, :, :])
        
        return m, z, s
