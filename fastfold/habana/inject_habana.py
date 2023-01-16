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

from fastfold.habana.fastnn import EvoformerStack, ExtraMSAStack

#from fastfold.model.fastnn.embedders import TemplateEmbedder
#from fastfold.model.fastnn.embedders_multimer import TemplateEmbedderMultimer
#from fastfold.model.fastnn.ops import RecyclingEmbedder, InputEmbedder


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

    copy_linear(model_fast.left_projection, model_ori.linear_a_p)
    copy_linear(model_fast.right_projection, model_ori.linear_b_p)

    copy_linear(model_fast.left_gate, model_ori.linear_a_g)
    copy_linear(model_fast.right_gate, model_ori.linear_b_g)


def copy_triangle_att(model_fast, model_ori):
    copy_layernorm(model_fast.layernorm1, model_ori.layer_norm)
    model_fast.linear_b_weights = model_ori.linear.weight
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
    copy_layernorm(block_fast.msa.MSARowAttentionWithPairBias.layernormM,
                   block_ori.msa_att_row.layer_norm_m)
    copy_layernorm(block_fast.msa.MSARowAttentionWithPairBias.layernormZ,
                   block_ori.msa_att_row.layer_norm_z)

    copy_attention(block_fast.msa.MSARowAttentionWithPairBias.attention, block_ori.msa_att_row.mha)

    block_fast.msa.MSARowAttentionWithPairBias.linear_b_weights.copy_(
        block_ori.msa_att_row.linear_z.weight)

    block_fast.msa.MSARowAttentionWithPairBias.out_bias.copy_(
        block_ori.msa_att_row.mha.linear_o.bias)

    # MSAColumnAttention
    copy_layernorm(block_fast.msa.MSAColumnAttention.layernormM,
                   block_ori.msa_att_col._msa_att.layer_norm_m)

    copy_attention(block_fast.msa.MSAColumnAttention.attention, block_ori.msa_att_col._msa_att.mha)

    # MSATransition
    copy_transition(block_fast.msa.MSATransition, block_ori.core.msa_transition)

    # communication
    copy_layernorm(block_fast.communication.layernormM,
                   block_ori.core.outer_product_mean.layer_norm)
    copy_linear(block_fast.communication.linear_a, block_ori.core.outer_product_mean.linear_1)
    copy_linear(block_fast.communication.linear_b, block_ori.core.outer_product_mean.linear_2)
    copy_linear(block_fast.communication.o_linear, block_ori.core.outer_product_mean.linear_out)

    # pair_stack
    # TriangleMultiplicationOutgoing
    copy_triangle(block_fast.pair.TriangleMultiplicationOutgoing, block_ori.core.tri_mul_out)
    # TriangleMultiplicationIncoming
    copy_triangle(block_fast.pair.TriangleMultiplicationIncoming, block_ori.core.tri_mul_in)

    # TriangleAttentionStartingNode
    copy_triangle_att(block_fast.pair.TriangleAttentionStartingNode, block_ori.core.tri_att_start)
    copy_triangle_att(block_fast.pair.TriangleAttentionEndingNode, block_ori.core.tri_att_end)

    copy_transition(block_fast.pair.PairTransition, block_ori.core.pair_transition)


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
        block_ori.msa_att_row.linear_z.weight)

    block_fast.msa_stack.MSARowAttentionWithPairBias.out_bias.copy_(
        block_ori.msa_att_row.mha.linear_o.bias)

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
        block_ori.core.
        outer_product_mean  # if not block_ori.is_multimer else block_ori.outer_product_mean
    )
    copy_layernorm(block_fast.communication.layernormM, comm_model.layer_norm)
    copy_linear(block_fast.communication.linear_a, comm_model.linear_1)
    copy_linear(block_fast.communication.linear_b, comm_model.linear_2)
    copy_linear(block_fast.communication.o_linear, comm_model.linear_out)

    # pair_stack
    # TriangleMultiplicationOutgoing
    copy_triangle(block_fast.pair_stack.TriangleMultiplicationOutgoing, block_ori.core.tri_mul_out)
    # TriangleMultiplicationIncoming
    copy_triangle(block_fast.pair_stack.TriangleMultiplicationIncoming, block_ori.core.tri_mul_in)

    # TriangleAttentionStartingNode
    copy_triangle_att(
        block_fast.pair_stack.TriangleAttentionStartingNode,
        block_ori.core.tri_att_start,
    )
    copy_triangle_att(block_fast.pair_stack.TriangleAttentionEndingNode, block_ori.core.tri_att_end)

    copy_transition(block_fast.pair_stack.PairTransition, block_ori.core.pair_transition)


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
    copy_linear(block_fast.template_pair_embedder.linear, block_ori.template_pair_embedder.linear)

    # TemplatePairStack
    copy_template_pair_block_para(block_fast.template_pair_stack, block_ori.template_pair_stack)
    copy_layernorm(block_fast.template_pair_stack.layer_norm,
                   block_ori.template_pair_stack.layer_norm)

    # TemplatePointwiseAttention
    copy_native_att(block_fast.template_pointwise_att.mha, block_ori.template_pointwise_att.mha)


def copy_template_multimer_para(block_fast, block_ori):
    # TemplatePairEmbedderMultimer
    copy_linear(block_fast.template_pair_embedder.dgram_linear,
                block_ori.template_pair_embedder.dgram_linear)
    copy_linear(block_fast.template_pair_embedder.aatype_linear_1,
                block_ori.template_pair_embedder.aatype_linear_1)
    copy_linear(block_fast.template_pair_embedder.aatype_linear_2,
                block_ori.template_pair_embedder.aatype_linear_2)
    copy_layernorm(block_fast.template_pair_embedder.query_embedding_layer_norm,
                   block_ori.template_pair_embedder.query_embedding_layer_norm)
    copy_linear(block_fast.template_pair_embedder.query_embedding_linear,
                block_ori.template_pair_embedder.query_embedding_linear)
    copy_linear(block_fast.template_pair_embedder.pseudo_beta_mask_linear,
                block_ori.template_pair_embedder.pseudo_beta_mask_linear)
    copy_linear(block_fast.template_pair_embedder.x_linear,
                block_ori.template_pair_embedder.x_linear)
    copy_linear(block_fast.template_pair_embedder.y_linear,
                block_ori.template_pair_embedder.y_linear)
    copy_linear(block_fast.template_pair_embedder.z_linear,
                block_ori.template_pair_embedder.z_linear)
    copy_linear(block_fast.template_pair_embedder.backbone_mask_linear,
                block_ori.template_pair_embedder.backbone_mask_linear)

    # TemplateSingleEmbedderMultimer
    copy_linear(block_fast.template_single_embedder.template_single_embedder,
                block_ori.template_single_embedder.template_single_embedder)
    copy_linear(block_fast.template_single_embedder.template_projector,
                block_ori.template_single_embedder.template_projector)

    # TemplatePairStack
    copy_template_pair_block_para(block_fast.template_pair_stack, block_ori.template_pair_stack)
    copy_layernorm(block_fast.template_pair_stack.layer_norm,
                   block_ori.template_pair_stack.layer_norm)

    # linear_t
    copy_linear(block_fast.linear_t, block_ori.linear_t)


def inject_evoformer(model):
    with torch.no_grad():
        target_module = model.evoformer
        fast_module = EvoformerStack(
            c_m=target_module.blocks[0].msa_att_row.c_in,
            c_z=target_module.blocks[0].msa_att_row.c_z,
            c_s=target_module.linear.out_features,
            no_blocks=len(target_module.blocks),
            blocks_per_ckpt=target_module.blocks_per_ckpt,
            clear_cache_between_blocks=target_module.clear_cache_between_blocks,
            is_multimer=target_module.blocks[0].is_multimer,
        )
        for target_block, fast_block in zip(target_module.blocks, fast_module.blocks):
            copy_evoformer_para(fast_block, target_block)
        if target_module.training == False:
            fast_module.eval()
        copy_linear(fast_module.linear, target_module.linear)
        model.evoformer = fast_module


def inject_extramsa(model):
    with torch.no_grad():
        target_module = model.extra_msa_stack
        fast_module = ExtraMSAStack(
            c_m=target_module.blocks[0].msa_att_row.c_in,
            c_z=target_module.blocks[0].msa_att_row.c_z,
            no_blocks=len(target_module.blocks),
            blocks_per_ckpt=1,
            clear_cache_between_blocks=target_module.clear_cache_between_blocks,
            is_multimer=target_module.blocks[0].is_multimer,
        )
        for target_block, fast_block in zip(target_module.blocks, fast_module.blocks):
            copy_extra_msa_para(fast_block, target_block)
        if target_module.training == False:
            fast_module.eval()
        model.extra_msa_stack = fast_module


def inject_template(model):
    with torch.no_grad():
        if model.evoformer.blocks[0].is_multimer:
            target_module = model.template_embedder
            fast_module = TemplateEmbedderMultimer(config=model.template_embedder.config)
            copy_template_multimer_para(fast_module, target_module)
            if target_module.training == False:
                fast_module.eval()
            model.template_embedder = fast_module
        else:
            target_module = model.template_embedder
            fast_module = TemplateEmbedder(config=model.template_embedder.config)
            copy_template_para(fast_module, target_module)
            if target_module.training == False:
                fast_module.eval()
            model.template_embedder = fast_module


def inject_embedder(model):
    if model.evoformer.blocks[0].is_multimer:
        return

    # recycle embedder
    with torch.no_grad():
        target_module = model.recycling_embedder
        fast_module = RecyclingEmbedder(c_m=target_module.c_m,
                                        c_z=target_module.c_z,
                                        min_bin=target_module.min_bin,
                                        max_bin=target_module.max_bin,
                                        no_bins=target_module.no_bins,
                                        inf=target_module.inf)
        copy_native_linear(fast_module.linear, target_module.linear)
        copy_layernorm(fast_module.layer_norm_m, target_module.layer_norm_m)
        copy_layernorm(fast_module.layer_norm_z, target_module.layer_norm_z)
        if target_module.training == False:
            fast_module.eval()
        model.recycling_embedder = fast_module

    # input embedder
    with torch.no_grad():
        target_module = model.input_embedder
        fast_module = InputEmbedder(
            tf_dim=target_module.tf_dim,
            msa_dim=target_module.msa_dim,
            c_z=target_module.c_z,
            c_m=target_module.c_m,
            relpos_k=target_module.relpos_k,
        )
        copy_linear(fast_module.linear_tf_z_i, target_module.linear_tf_z_i)
        copy_linear(fast_module.linear_tf_z_j, target_module.linear_tf_z_j)
        copy_linear(fast_module.linear_tf_m, target_module.linear_tf_m)
        copy_linear(fast_module.linear_msa_m, target_module.linear_msa_m)
        copy_linear(fast_module.linear_relpos, target_module.linear_relpos)
        if target_module.training == False:
            fast_module.eval()
        model.input_embedder = fast_module


def inject_habana(model):
    inject_evoformer(model)
    inject_extramsa(model)
    #inject_template(model)
    #inject_embedder(model)
    return model
