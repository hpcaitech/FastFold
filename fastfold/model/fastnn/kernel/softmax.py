from functools import reduce
from operator import mul
import logging

import torch

_triton_available = True
if _triton_available:
    try:
        from .triton.softmax import softmax_triton_kernel_wrapper
        from .triton.softmax import softmax_grad_triton_kernel_wrapper

    except ImportError:
        logging.warning("Triton is not available, fallback to old kernel.")
        _triton_available = False

from .cuda_native.softmax import softmax_cuda_kernel_wrapper
from .cuda_native.softmax import softmax_grad_cuda_kernel_wrapper


class FusedSoftmaxFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, mask=None, bias=None):
        input_ = input.contiguous()
        mask_, bias_ = None, None
        ctx.use_bias = False
        if mask is not None:
            mask_ = mask.contiguous()
        if bias is not None:
            bias_ = bias.contiguous()
            ctx.use_bias = True
        ctx.cols = input_.shape[-1]
        ctx.rows = reduce(mul, input.shape[:-1])
        if _triton_available:
            output = softmax_triton_kernel_wrapper(input_, mask_, bias_, ctx.rows, ctx.cols)
        else:
            output = softmax_cuda_kernel_wrapper(input_, mask_, bias_, ctx.rows, ctx.cols)
        ctx.save_for_backward(output, mask_)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        output, mask_ = ctx.saved_tensors
        if _triton_available:
            grad_input = softmax_grad_triton_kernel_wrapper(grad_output, output, ctx.rows, ctx.cols)
        else:
            grad_input = softmax_grad_cuda_kernel_wrapper(grad_output, output, mask_, ctx.rows,
                                                          ctx.cols)
        grad_bias = None
        if ctx.use_bias:
            grad_bias = torch.sum(grad_input, dim=1, keepdim=True)

        return grad_input, None, grad_bias


fused_softmax = FusedSoftmaxFunc.apply