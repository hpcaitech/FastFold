from functools import partial
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from fastfold.habana.distributed import All_to_All, gather, scatter
from fastfold.utils.checkpointing import checkpoint_blocks

from .msa import ExtraMSACore, MSAStack
from .ops import Linear, OutProductMean
from .triangle import PairStack

import habana_frameworks.torch.core as htcore

class Evoformer(nn.Module):

    def __init__(self,
                 c_m: int,
                 c_z: int,
                 first_block: bool,
                 last_block: bool,
                 is_multimer: bool = False):
        super(Evoformer, self).__init__()

        self.first_block = first_block
        self.last_block = last_block

        self.msa = MSAStack(c_m, c_z, p_drop=0.15)
        self.communication = OutProductMean(n_feat=c_m, n_feat_out=c_z, n_feat_proj=32)
        self.pair = PairStack(d_pair=c_z)
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

        dap_size = dist.get_world_size()

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

        # msa_mask = msa_mask.unsqueeze(0)
        # pair_mask = pair_mask.unsqueeze(0)

        msa_mask = torch.nn.functional.pad(msa_mask, (0, padding_size))
        pair_mask = torch.nn.functional.pad(pair_mask, (0, padding_size, 0, padding_size))

        if not self.is_multimer:
            m = self.msa(m, z, msa_mask)
            z = self.communication(m, msa_mask, z)
            m = All_to_All.apply(m, 1, 2)
            z = self.pair(z, pair_mask)
        else:
            z = self.communication(m, msa_mask, z)
            z_ori = z
            m = All_to_All.apply(m, 1, 2)
            z = self.pair(z, pair_mask)
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

        htcore.mark_step()

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

    def forward(
        self,
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

        msa_mask = msa_mask.unsqueeze(0)
        pair_mask = pair_mask.unsqueeze(0)

        blocks = [
            partial(
                b,
                msa_mask=msa_mask,
                pair_mask=pair_mask,
                chunk_size=chunk_size,
                _mask_trans=_mask_trans,
            ) for b in self.blocks
        ]

        if torch.is_grad_enabled():
            m, z = checkpoint_blocks(
                blocks,
                args=(m, z),
                blocks_per_ckpt=self.blocks_per_ckpt if self.training else None,
            )
        else:
            for b in blocks:
                m, z = b(m, z)

        s = self.linear(m[..., 0, :, :])

        htcore.mark_step()

        return m, z, s


class ExtraMSABlock(nn.Module):

    def __init__(self,
                 c_m: int,
                 c_z: int,
                 first_block: bool,
                 last_block: bool,
                 is_multimer: bool = False):
        super(ExtraMSABlock, self).__init__()

        self.first_block = first_block
        self.last_block = last_block

        self.msa_stack = ExtraMSACore(c_m, c_z, p_drop=0.15)
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

        htcore.mark_step()

        dap_size = dist.get_world_size()

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
            m = self.msa_stack(m, z, msa_mask)
            z = self.communication(m, msa_mask, z)
            m = All_to_All.apply(m, 1, 2)
            z = self.pair_stack(z, pair_mask)
        else:
            z = self.communication(m, msa_mask, z)
            z_ori = z
            m = All_to_All.apply(m, 1, 2)
            z = self.pair_stack(z, pair_mask)
            m = self.msa_stack(m, z_ori, msa_mask)

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

        htcore.mark_step()

        return m, z


class ExtraMSAStack(nn.Module):
    """
    Implements Algorithm 18.
    """

    def __init__(
        self,
        c_m: int,
        c_z: int,
        no_blocks: int,
        blocks_per_ckpt: int,
        clear_cache_between_blocks: bool = False,
        is_multimer: bool = False,
        **kwargs,
    ):
        super(ExtraMSAStack, self).__init__()

        self.blocks_per_ckpt = blocks_per_ckpt

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

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        chunk_size: int,
        msa_mask: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None,
        _mask_trans: bool = True,
    ) -> torch.Tensor:
        blocks = [
            partial(
                b,
                msa_mask=msa_mask,
                pair_mask=pair_mask,
                chunk_size=chunk_size,
                _mask_trans=_mask_trans,
            ) for b in self.blocks
        ]

        if torch.is_grad_enabled():
            m, z = checkpoint_blocks(
                blocks,
                args=(m, z),
                blocks_per_ckpt=self.blocks_per_ckpt if self.training else None,
            )
        else:
            for b in blocks:
                m, z = b(m, z)

        return z
