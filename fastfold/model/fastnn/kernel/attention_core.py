import math

import torch
from einops import rearrange

import triton
from .triton.attention_core import _attention_core


class FusedAttenionCoreFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, mask=None, bias=None):

        assert (q.dtype
                in [torch.float16,
                    torch.bfloat16]), "triton flash attention only support float16/bfloat16 now"

        q_ori_size = list(q.size())

        batch = q_ori_size[0]

        if len(q_ori_size) == 5:
            q = rearrange(q, 'b1 b2 h n d -> (b1 b2) h n d')
            k = rearrange(k, 'b1 b2 h n d -> (b1 b2) h n d')
            v = rearrange(v, 'b1 b2 h n d -> (b1 b2) h n d')

        sm_scale = 1. / math.sqrt(q.size(-1))
        # q *= sm_scale
        BLOCK = 128
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        o = torch.empty_like(q)
        grid = (triton.cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1])
        tmp = torch.empty((q.shape[0] * q.shape[1], q.shape[2]),
                          device=q.device,
                          dtype=torch.float32)
        num_warps = 4 if Lk <= 64 else 8

        _attention_core[grid](
            q,
            k,
            v,
            mask,
            bias,
            sm_scale,
            tmp,
            o,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            q.shape[0],
            q.shape[1],
            q.shape[2],
            batch,
            BLOCK_M=BLOCK,
            BLOCK_N=BLOCK,
            BLOCK_DMODEL=Lk,
            use_mask=(mask != None),
            use_bias=(bias != None),
            num_warps=num_warps,
            num_stages=1,
        )

        if len(q_ori_size) == 5:
            o = rearrange(o, '(b1 b2) h n d -> b1 b2 n (h d)', b1=batch)

        # ctx.save_for_backward(q, k, v, o, L, m, mask, bias)
        # ctx.BLOCK = BLOCK
        # ctx.grid = grid
        # ctx.sm_scale = sm_scale
        # ctx.BLOCK_DMODEL = Lk

        return o


fused_attention_core = FusedAttenionCoreFunc.apply