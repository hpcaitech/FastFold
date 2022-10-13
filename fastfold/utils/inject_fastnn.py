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
import fastfold

from fastfold.model.fastnn import EvoformerBlock, ExtraMSABlock, TemplatePairStackBlock
from fastfold.model.fastnn.embedders import TemplateEmbedder

def copy_layernorm(model_fast, model_ori):
    model_fast.weight.copy_(model_ori.weight)
    model_fast.bias.copy_(model_ori.bias)


def copy_linear(model_fast, model_ori):
    model_fast.weight.copy_(model_ori.weight)
    if model_fast.use_bias:
        model_fast.bias.copy_(model_ori.bias)


def copy_native_linear(model_fast, model_ori):
    model_fast.weight.copy_(model_ori.weight)
    try:
        model_fast.bias.copy_(model_ori.bias)
    except:
        pass


def copy_kv_linear(model_fast, ori_k, ori_v):
    model_fast.weight.copy_(torch.cat((ori_k.weight, ori_v.weight), dim=0))


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


def copy_native_att(model_fast, model_ori):
    copy_native_linear(model_fast.linear_q, model_ori.linear_q)
    copy_native_linear(model_fast.linear_k, model_ori.linear_k)
    copy_native_linear(model_fast.linear_v, model_ori.linear_v)
    copy_native_linear(model_fast.linear_o, model_ori.linear_o)
    if model_ori.gating:
         copy_native_linear(model_fast.linear_g, model_ori.linear_g)


def copy_evoformer_para(block_fast, block_ori):
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


def copy_global_attention(model_fast, model_ori):
    copy_linear(model_fast.to_q, model_ori.linear_q)
    copy_kv_linear(model_fast.to_kv, model_ori.linear_k, model_ori.linear_v)
    copy_linear(model_fast.gating_linear, model_ori.linear_g)
    copy_linear(model_fast.o_linear, model_ori.linear_o)

    try:
        model_fast.gating_bias.copy_(model_ori.linear_g.bias)
    except:
        print("no gating_bias need copy")


def copy_extra_msa_para(block_fast, block_ori):
    # msa_stack
    # MSARowAttentionWithPairBias
    copy_layernorm(
        block_fast.msa_stack.MSARowAttentionWithPairBias.layernormM,
        block_ori.msa_att_row.layer_norm_m,
    )
    copy_layernorm(
        block_fast.msa_stack.MSARowAttentionWithPairBias.layernormZ,
        block_ori.msa_att_row.layer_norm_z,
    )

    copy_attention(
        block_fast.msa_stack.MSARowAttentionWithPairBias.attention,
        block_ori.msa_att_row.mha,
    )

    block_fast.msa_stack.MSARowAttentionWithPairBias.linear_b_weights.copy_(
        block_ori.msa_att_row.linear_z.weight
    )

    block_fast.msa_stack.MSARowAttentionWithPairBias.out_bias.copy_(
        block_ori.msa_att_row.mha.linear_o.bias
    )

    # MSAColumnAttention
    copy_layernorm(
        block_fast.msa_stack.MSAColumnAttention.layernormM,
        block_ori.msa_att_col.layer_norm_m,
    )

    copy_global_attention(
        block_fast.msa_stack.MSAColumnAttention.global_attention,
        block_ori.msa_att_col.global_attention,
    )

    # MSATransition
    copy_transition(block_fast.msa_stack.MSATransition, block_ori.core.msa_transition)

    # communication
    comm_model = (
        block_ori.core.outer_product_mean# if not block_ori.is_multimer else block_ori.outer_product_mean
    )
    copy_layernorm(block_fast.communication.layernormM, comm_model.layer_norm)
    copy_linear(block_fast.communication.linear_a, comm_model.linear_1)
    copy_linear(block_fast.communication.linear_b, comm_model.linear_2)
    copy_linear(block_fast.communication.o_linear, comm_model.linear_out)

    # pair_stack
    # TriangleMultiplicationOutgoing
    copy_triangle(
        block_fast.pair_stack.TriangleMultiplicationOutgoing, block_ori.core.tri_mul_out
    )
    # TriangleMultiplicationIncoming
    copy_triangle(
        block_fast.pair_stack.TriangleMultiplicationIncoming, block_ori.core.tri_mul_in
    )

    # TriangleAttentionStartingNode
    copy_triangle_att(
        block_fast.pair_stack.TriangleAttentionStartingNode,
        block_ori.core.tri_att_start,
    )
    copy_triangle_att(
        block_fast.pair_stack.TriangleAttentionEndingNode, block_ori.core.tri_att_end
    )

    copy_transition(
        block_fast.pair_stack.PairTransition, block_ori.core.pair_transition
    )


def copy_template_pair_stack_para(block_fast, block_ori):
    # TriangleMultiplicationOutgoing
    copy_triangle(block_fast.TriangleMultiplicationOutgoing, block_ori.tri_mul_out)
    # TriangleMultiplicationIncoming
    copy_triangle(block_fast.TriangleMultiplicationIncoming, block_ori.tri_mul_in)

    # TriangleAttentionStartingNode
    copy_triangle_att(block_fast.TriangleAttentionStartingNode, block_ori.tri_att_start)
    copy_triangle_att(block_fast.TriangleAttentionEndingNode, block_ori.tri_att_end)

    copy_transition(block_fast.PairTransition, block_ori.pair_transition)


def copy_template_pair_block_para(fast_module, target_module):
    with torch.no_grad():
        for ori_block, fast_block in zip(target_module.blocks, fast_module.blocks):
            copy_template_pair_stack_para(fast_block, ori_block)
            if ori_block.training == False:
                fast_block.eval()


def copy_template_para(block_fast, block_ori):
    # TemplateAngleEmbedder
    copy_linear(block_fast.template_angle_embedder.linear_1, 
                block_ori.template_angle_embedder.linear_1)
    copy_linear(block_fast.template_angle_embedder.linear_2, 
                block_ori.template_angle_embedder.linear_2)
    
    # TemplatePairEmbedder
    copy_linear(block_fast.template_pair_embedder.linear, 
                block_ori.template_pair_embedder.linear)
    
    # TemplatePairStack
    copy_template_pair_block_para(block_fast.template_pair_stack, 
                                  block_ori.template_pair_stack)
    copy_layernorm(block_fast.template_pair_stack.layer_norm,
                   block_ori.template_pair_stack.layer_norm)
    
    # TemplatePointwiseAttention
    copy_native_att(block_fast.template_pointwise_att.mha,
                    block_ori.template_pointwise_att.mha)


def inject_evoformer(model):
    with torch.no_grad():
        fastfold_blocks = nn.ModuleList()
        for block_id, ori_block in enumerate(model.evoformer.blocks):
            c_m = ori_block.msa_att_row.c_in
            c_z = ori_block.msa_att_row.c_z
            is_multimer = ori_block.is_multimer
            fastfold_block = EvoformerBlock(c_m=c_m,
                                            c_z=c_z,
                                            first_block=(block_id == 0),
                                            last_block=(block_id == len(model.evoformer.blocks) - 1),
                                            is_multimer=is_multimer,
                                        )

            copy_evoformer_para(fastfold_block, ori_block)

            fastfold_blocks.append(fastfold_block)

        model.evoformer.blocks = fastfold_blocks

    return model


def inject_extraMsaBlock(model):
    with torch.no_grad():
        new_model_blocks = nn.ModuleList()
        for block_id, ori_block in enumerate(model.extra_msa_stack.blocks):
            c_m = ori_block.msa_att_row.c_in
            c_z = ori_block.msa_att_row.c_z
            is_multimer = ori_block.is_multimer
            new_model_block = ExtraMSABlock(
                c_m=c_m,
                c_z=c_z,
                first_block=(block_id == 0),
                last_block=(block_id == len(model.extra_msa_stack.blocks) - 1),
                is_multimer=is_multimer
            )

            copy_extra_msa_para(new_model_block, ori_block)
            if ori_block.training == False:
                new_model_block.eval()
            new_model_blocks.append(new_model_block)

        model.extra_msa_stack.blocks = new_model_blocks


def inject_template(model):
    with torch.no_grad():
        target_module = model.template_embedder
        fast_module = TemplateEmbedder(config=model.template_embedder.config)
        copy_template_para(fast_module, target_module)
        if target_module.training == False:
            fast_module.eval()
        model.template_embedder = fast_module


def inject_fastnn(model):
    inject_evoformer(model)
    inject_extraMsaBlock(model)
    inject_template(model)
    return model