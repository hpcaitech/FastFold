from typing import Optional, Tuple
from functools import partial

import torch
import torch.nn as nn
from fastfold.utils.checkpointing import checkpoint_blocks

from .msa import MSAStack
from .ops import OutProductMean, Linear
from .triangle import PairStack

class Evoformer(nn.Module):

    def __init__(self, c_m: int, c_z: int, first_block: bool, last_block: bool, is_multimer: bool=False):
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

        if self.first_block:
            m = m.unsqueeze(0)
            z = z.unsqueeze(0)

        msa_mask = msa_mask.unsqueeze(0)
        pair_mask = pair_mask.unsqueeze(0)

        if not self.is_multimer:
            m = self.msa(m, z)
            z = self.communication(m)
            z = self.pair(z)
        else:
            z = self.communication(m)
            z_ori = z
            z = self.pair(z)
            m = self.msa(m, z_ori)

        if self.last_block:
            m = m.squeeze(0)
            z = z.squeeze(0)

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

        m, z = checkpoint_blocks(
            blocks,
            args=(m, z),
            blocks_per_ckpt=self.blocks_per_ckpt if self.training else None,
        )

        s = self.linear(m[..., 0, :, :])
        
        return m, z, s
