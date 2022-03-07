from typing import Tuple, Optional

import torch
import torch.nn as nn

from fastfold.model import MSAStack, OutProductMean, PairStack
from fastfold.distributed.comm_async import All_to_All_Async, All_to_All_Async_Opp
from fastfold.distributed.comm import gather, scatter


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

        if self.first_block:
            m = m.unsqueeze(0)
            z = z.unsqueeze(0)

            m = scatter(m, dim=1)
            z = scatter(z, dim=1)

        msa_mask = msa_mask.unsqueeze(0)
        pair_mask = pair_mask.unsqueeze(0)

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

        return m, z


def copy_layernorm(model_fast, model_ori):
    model_fast.weight.copy_(model_ori.weight)
    model_fast.bias.copy_(model_ori.bias)


def copy_linear(model_fast, model_ori):
    model_fast.weight.copy_(model_ori.weight)
    if model_fast.use_bias:
        model_fast.bias.copy_(model_ori.bias)


def copy_qkv_linear(model_fast, ori_q, ori_k, ori_v):
    model_fast.weight.copy_(torch.cat((ori_q.weight, ori_k.weight, ori_v.weight), dim=0))


def copy_attention(model_fast, model_ori):
    copy_qkv_linear(model_fast.to_qkv, model_ori.linear_q, model_ori.linear_k, model_ori.linear_v)
    copy_linear(model_fast.gating_linear, model_ori.linear_g)
    copy_linear(model_fast.o_linear, model_ori.linear_o)

    try:
        model_fast.gating_bias.copy_(model_ori.linear_g.bias)
    except:
        print("no gating_bias need copy")


def copy_left_right(model_fast, ori_left, ori_right):
    model_fast.weight.copy_(torch.cat((ori_left.weight, ori_right.weight), dim=0))
    model_fast.bias.copy_(torch.cat((ori_left.bias, ori_right.bias), dim=0))


def copy_transition(model_fast, model_ori):
    copy_layernorm(model_fast.norm, model_ori.layer_norm)
    copy_linear(model_fast.linear1, model_ori.linear_1)
    copy_linear(model_fast.linear2, model_ori.linear_2)


def copy_triangle(model_fast, model_ori):
    copy_layernorm(model_fast.layernorm1, model_ori.layer_norm_in)
    copy_layernorm(model_fast.layernorm2, model_ori.layer_norm_out)
    copy_linear(model_fast.output_gate, model_ori.linear_g)
    copy_linear(model_fast.output_projection, model_ori.linear_z)
    model_fast.output_bias.copy_(model_ori.linear_z.bias)

    copy_left_right(model_fast.left_right_projection, model_ori.linear_a_p, model_ori.linear_b_p)

    copy_left_right(model_fast.left_right_gate, model_ori.linear_a_g, model_ori.linear_b_g)


def copy_triangle_att(model_fast, model_ori):
    copy_layernorm(model_fast.layernorm1, model_ori.layer_norm)
    copy_linear(model_fast.linear_b, model_ori.linear)
    copy_attention(model_fast.attention, model_ori.mha)

    model_fast.out_bias.copy_(model_ori.mha.linear_o.bias)


def copy_para(block_fast, block_ori):
    # msa_stack
    # MSARowAttentionWithPairBias
    copy_layernorm(block_fast.msa_stack.MSARowAttentionWithPairBias.layernormM,
                   block_ori.msa_att_row.layer_norm_m)
    copy_layernorm(block_fast.msa_stack.MSARowAttentionWithPairBias.layernormZ,
                   block_ori.msa_att_row.layer_norm_z)

    copy_attention(block_fast.msa_stack.MSARowAttentionWithPairBias.attention,
                   block_ori.msa_att_row.mha)

    block_fast.msa_stack.MSARowAttentionWithPairBias.linear_b_weights.copy_(
        block_ori.msa_att_row.linear_z.weight)

    block_fast.msa_stack.MSARowAttentionWithPairBias.out_bias.copy_(
        block_ori.msa_att_row.mha.linear_o.bias)

    # MSAColumnAttention
    copy_layernorm(block_fast.msa_stack.MSAColumnAttention.layernormM,
                   block_ori.msa_att_col._msa_att.layer_norm_m)

    copy_attention(block_fast.msa_stack.MSAColumnAttention.attention,
                   block_ori.msa_att_col._msa_att.mha)

    # MSATransition
    copy_transition(block_fast.msa_stack.MSATransition, block_ori.core.msa_transition)

    # communication
    copy_layernorm(block_fast.communication.layernormM,
                   block_ori.core.outer_product_mean.layer_norm)
    copy_linear(block_fast.communication.linear_a, block_ori.core.outer_product_mean.linear_1)
    copy_linear(block_fast.communication.linear_b, block_ori.core.outer_product_mean.linear_2)
    copy_linear(block_fast.communication.o_linear, block_ori.core.outer_product_mean.linear_out)

    # pair_stack
    # TriangleMultiplicationOutgoing
    copy_triangle(block_fast.pair_stack.TriangleMultiplicationOutgoing, block_ori.core.tri_mul_out)
    # TriangleMultiplicationIncoming
    copy_triangle(block_fast.pair_stack.TriangleMultiplicationIncoming, block_ori.core.tri_mul_in)

    # TriangleAttentionStartingNode
    copy_triangle_att(block_fast.pair_stack.TriangleAttentionStartingNode,
                      block_ori.core.tri_att_start)
    copy_triangle_att(block_fast.pair_stack.TriangleAttentionEndingNode, block_ori.core.tri_att_end)

    copy_transition(block_fast.pair_stack.PairTransition, block_ori.core.pair_transition)


def inject_openfold(model):
    with torch.no_grad():
        fastfold_blocks = nn.ModuleList()
        for block_id, openfold_block in enumerate(model.evoformer.blocks):
            c_m = openfold_block.msa_att_row.c_in
            c_z = openfold_block.msa_att_row.c_z
            fastfold_block = EvoformerBlock(c_m=c_m,
                                            c_z=c_z,
                                            first_block=(block_id == 0),
                                            last_block=(block_id == len(model.evoformer.blocks) -
                                                        1))

            copy_para(fastfold_block, openfold_block)

            fastfold_blocks.append(fastfold_block)

        model.evoformer.blocks = fastfold_blocks

    return model
