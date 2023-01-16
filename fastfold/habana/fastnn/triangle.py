import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn import LayerNorm

from fastfold.habana.distributed import col_to_row, gather, row_to_col, scatter

from .kernel import bias_dropout_add, bias_ele_dropout_residual
from .ops import Linear, SelfAttention, Transition


def permute_final_dims(tensor, inds):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


class TriangleMultiplicationOutgoing(nn.Module):

    def __init__(self, d_pair, p_drop, c=128):
        super(TriangleMultiplicationOutgoing, self).__init__()
        self.d_pair = d_pair
        self.c = c

        self.layernorm1 = LayerNorm(d_pair)
        self.left_projection = Linear(d_pair, c)
        self.right_projection = Linear(d_pair, c)
        self.left_gate = Linear(d_pair, c, initializer='zeros', bias_init=1.)
        self.right_gate = Linear(d_pair, c, initializer='zeros', bias_init=1.)

        self.output_gate = Linear(d_pair, d_pair, initializer='zeros', bias_init=1.)
        self.layernorm2 = LayerNorm(c)
        self.output_projection = Linear(d_pair, d_pair, initializer='zeros', use_bias=False)
        self.output_bias = nn.parameter.Parameter(data=torch.zeros((d_pair,)), requires_grad=True)

        self.p_drop = p_drop

    def forward(self, Z_raw, Z_mask):
        Z = self.layernorm1(Z_raw)
        left_proj_act = self.left_projection(Z)
        right_proj_act = self.right_projection(Z)
        left_proj_act = Z_mask.unsqueeze(-1) * left_proj_act
        right_proj_act = Z_mask.unsqueeze(-1) * right_proj_act

        left_proj_act *= torch.sigmoid(self.left_gate(Z))
        right_proj_act *= torch.sigmoid(self.right_gate(Z))

        right_proj_act = gather(right_proj_act.contiguous(), dim=1)

        g = torch.sigmoid(self.output_gate(Z))
        p = torch.matmul(
            permute_final_dims(left_proj_act, (2, 0, 1)),
            permute_final_dims(right_proj_act, (2, 1, 0)),
        )
        ab = permute_final_dims(p, (1, 2, 0))

        # ab = torch.einsum('bikd,bjkd->bijd', left_proj_act, right_proj_act)
        ab = self.output_projection(self.layernorm2(ab))
        dropout_mask = torch.ones_like(Z[:, 0:1, :, :], device=Z.device, dtype=Z.dtype)
        return bias_ele_dropout_residual(ab,
                                         self.output_bias,
                                         g,
                                         dropout_mask,
                                         Z_raw,
                                         prob=self.p_drop,
                                         training=self.training)


class TriangleMultiplicationIncoming(nn.Module):

    def __init__(self, d_pair, p_drop, c=128):
        super(TriangleMultiplicationIncoming, self).__init__()
        self.d_pair = d_pair
        self.c = c

        self.layernorm1 = LayerNorm(d_pair)
        self.left_projection = Linear(d_pair, c)
        self.right_projection = Linear(d_pair, c)
        self.left_gate = Linear(d_pair, c, initializer='zeros', bias_init=1.)
        self.right_gate = Linear(d_pair, c, initializer='zeros', bias_init=1.)

        self.output_gate = Linear(d_pair, d_pair, initializer='zeros', bias_init=1.)
        self.layernorm2 = LayerNorm(c)
        self.output_projection = Linear(d_pair, d_pair, initializer='zeros', use_bias=False)
        self.output_bias = nn.parameter.Parameter(data=torch.zeros((d_pair,)), requires_grad=True)

        self.p_drop = p_drop

    def forward(self, Z_raw, Z_mask):
        Z = self.layernorm1(Z_raw)
        left_proj_act = self.left_projection(Z)
        right_proj_act = self.right_projection(Z)
        left_proj_act = Z_mask.unsqueeze(-1) * left_proj_act
        right_proj_act = Z_mask.unsqueeze(-1) * right_proj_act

        left_proj_act *= torch.sigmoid(self.left_gate(Z))
        right_proj_act *= torch.sigmoid(self.right_gate(Z))

        left_proj_act = gather(left_proj_act.contiguous(), dim=2)

        g = torch.sigmoid(self.output_gate(Z))
        p = torch.matmul(
            permute_final_dims(left_proj_act, (2, 1, 0)),
            permute_final_dims(right_proj_act, (2, 0, 1)),
        )
        ab = permute_final_dims(p, (1, 2, 0))

        # ab = torch.einsum('bkid,bkjd->bijd', left_proj_act, right_proj_act)
        ab = self.output_projection(self.layernorm2(ab))
        dropout_mask = torch.ones_like(Z[:, 0:1, :, :], device=Z.device, dtype=Z.dtype)
        return bias_ele_dropout_residual(ab,
                                         self.output_bias,
                                         g,
                                         dropout_mask,
                                         Z_raw,
                                         prob=self.p_drop,
                                         training=self.training)


class TriangleAttentionStartingNode(nn.Module):

    def __init__(self, d_pair, p_drop, c=32, n_head=4):
        super(TriangleAttentionStartingNode, self).__init__()
        self.d_pair = d_pair
        self.c = c
        self.n_head = n_head
        self.p_drop = p_drop

        self.layernorm1 = LayerNorm(d_pair)
        _init_weights = torch.nn.init.normal_(torch.zeros([n_head, d_pair]),
                                              std=1.0 / math.sqrt(d_pair))
        self.linear_b_weights = nn.parameter.Parameter(data=_init_weights)
        self.attention = SelfAttention(qkv_dim=d_pair,
                                       c=c,
                                       n_head=n_head,
                                       out_dim=d_pair,
                                       gating=True,
                                       last_bias_fuse=True)

        self.out_bias = nn.parameter.Parameter(data=torch.zeros((d_pair,)), requires_grad=True)

    def forward(self, Z_raw, Z_mask):
        Z = self.layernorm1(Z_raw)
        b = F.linear(Z, self.linear_b_weights)
        b = gather(b, dim=1)
        b = rearrange(b, 'b q k h -> b h q k')

        Z = self.attention(Z, Z_mask, b)

        dropout_mask = torch.ones_like(Z[:, 0:1, :, :], device=Z.device, dtype=Z.dtype)
        return bias_dropout_add(Z,
                                self.out_bias,
                                dropout_mask,
                                Z_raw,
                                prob=self.p_drop,
                                training=self.training)


class TriangleAttentionEndingNode(nn.Module):

    def __init__(self, d_pair, p_drop, c=32, n_head=4):
        super(TriangleAttentionEndingNode, self).__init__()
        self.d_pair = d_pair
        self.c = c
        self.n_head = n_head
        self.p_drop = p_drop

        self.layernorm1 = LayerNorm(d_pair)
        _init_weights = torch.nn.init.normal_(torch.zeros([n_head, d_pair]),
                                              std=1.0 / math.sqrt(d_pair))
        self.linear_b_weights = nn.parameter.Parameter(data=_init_weights)
        self.attention = SelfAttention(qkv_dim=d_pair,
                                       c=c,
                                       n_head=n_head,
                                       out_dim=d_pair,
                                       gating=True,
                                       last_bias_fuse=True)

        self.out_bias = nn.parameter.Parameter(data=torch.zeros((d_pair,)), requires_grad=True)

    def forward(self, Z_raw, Z_mask):
        Z = Z_raw.transpose(-2, -3)
        Z_mask = Z_mask.transpose(-1, -2)

        Z = self.layernorm1(Z)
        b = F.linear(Z, self.linear_b_weights)
        b = gather(b, dim=1)
        b = rearrange(b, 'b q k h -> b h q k')

        Z = self.attention(Z, Z_mask, b)

        Z = Z.transpose(-2, -3)
        dropout_mask = torch.ones_like(Z[:, :, 0:1, :], device=Z.device, dtype=Z.dtype)
        return bias_dropout_add(Z,
                                self.out_bias,
                                dropout_mask,
                                Z_raw,
                                prob=self.p_drop,
                                training=self.training)


class PairStack(nn.Module):

    def __init__(self, d_pair, p_drop=0.25):
        super(PairStack, self).__init__()

        self.TriangleMultiplicationOutgoing = TriangleMultiplicationOutgoing(d_pair, p_drop=p_drop)
        self.TriangleMultiplicationIncoming = TriangleMultiplicationIncoming(d_pair, p_drop=p_drop)
        self.TriangleAttentionStartingNode = TriangleAttentionStartingNode(d_pair, p_drop=p_drop)
        self.TriangleAttentionEndingNode = TriangleAttentionEndingNode(d_pair, p_drop=p_drop)
        self.PairTransition = Transition(d=d_pair)

    def forward(self, pair, pair_mask):
        pair_mask_row = scatter(pair_mask, dim=1)
        pair_mask_col = scatter(pair_mask, dim=2)
        pair = self.TriangleMultiplicationOutgoing(pair, pair_mask_row)
        pair = row_to_col(pair)
        pair = self.TriangleMultiplicationIncoming(pair, pair_mask_col)
        pair = col_to_row(pair)
        pair = self.TriangleAttentionStartingNode(pair, pair_mask_row)
        pair = row_to_col(pair)
        pair = self.TriangleAttentionEndingNode(pair, pair_mask_col)
        pair = self.PairTransition(pair)
        pair = col_to_row(pair)
        return pair
