import math
import logging

import torch
from einops import rearrange

_triton_available = True
if _triton_available:
    try:
        from .triton.attention_core import attention_core_triton_kernel_wrapper

    except ImportError:
        logging.warning("Triton is not available, fallback to old kernel.")
        _triton_available = False


def _torch_attention_core(q, k, v, mask, bias):
    scaling = 1. / math.sqrt(q.size(-1))
    q = q * scaling

    logits = torch.matmul(q, k.transpose(-1, -2))
    logits += bias
    logits += (1e20 * (mask - 1))[..., :, None, None, :]

    weights = torch.nn.functional.softmax(logits.float(), -1).to(dtype=q.dtype)

    weighted_avg = torch.matmul(weights, v)

    weighted_avg = rearrange(weighted_avg, 'b1 b2 h n d -> b1 b2 n (h d)')

    return weighted_avg


class FusedAttenionCoreFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, mask=None, bias=None):

        if _triton_available:
            o = attention_core_triton_kernel_wrapper(q, k, v, mask, bias)
        else:
            o = _torch_attention_core(q, k, v, mask, bias)

        # ctx.save_for_backward(q, k, v, o, L, m, mask, bias)
        # ctx.BLOCK = BLOCK
        # ctx.grid = grid
        # ctx.sm_scale = sm_scale
        # ctx.BLOCK_DMODEL = Lk

        return o


fused_attention_core = FusedAttenionCoreFunc.apply