import importlib
from functools import reduce
from operator import mul

import torch

fastfold_softmax_cuda = importlib.import_module("fastfold_softmax_cuda")


class SoftmaxAffineFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        input_ = input.contiguous()
        ctx.cols = input_.shape[-1]
        ctx.rows = reduce(mul, input.shape[:-1])
        output = fastfold_softmax_cuda.forward(input_, ctx.rows, ctx.cols)
        ctx.save_for_backward(output)

        return output

    @staticmethod
    def backward(ctx, grad_output):

        output = ctx.saved_tensors[0]

        grad_input = None
        grad_input = fastfold_softmax_cuda.backward(grad_output.contiguous(), output,
                                                           ctx.rows, ctx.cols)

        return grad_input


class FusedMaskSoftmaxFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, mask):
        input_ = input.contiguous()
        mask_ = mask.contiguous()
        ctx.cols = input_.shape[-1]
        ctx.rows = reduce(mul, input.shape[:-1])
        output = fastfold_softmax_cuda.fused_mask_softmax_forward(
            input_, mask_, ctx.rows, ctx.cols)
        ctx.save_for_backward(output, mask_)

        return output

    @staticmethod
    def backward(ctx, grad_output):

        output, mask_ = ctx.saved_tensors

        grad_input = None
        grad_input = fastfold_softmax_cuda.fused_mask_softmax_backward(
            grad_output.contiguous(), output, mask_, ctx.rows, ctx.cols)

        return grad_input.contiguous(), None, None


class FusedMaskBiasSoftmaxFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, mask, bias):
        input_ = input.contiguous()
        mask_ = mask.contiguous()
        bias_ = bias.contiguous()
        ctx.cols = input_.shape[-1]
        ctx.rows = reduce(mul, input.shape[:-1])
        output = fastfold_softmax_cuda.fused_mask_bias_softmax_forward(
            input_, mask_, bias_, ctx.rows, ctx.cols)
        ctx.save_for_backward(output, mask_, bias_)

        return output

    @staticmethod
    def backward(ctx, grad_output):

        output, mask_, bias_ = ctx.saved_tensors

        grad_input = None
        grad_input = fastfold_softmax_cuda.fused_mask_bias_softmax_backward(
            grad_output.contiguous(), output, mask_, bias_, ctx.rows, ctx.cols)

        grad_input = grad_input.contiguous()

        grad_bias = torch.sum(grad_input, dim=1, keepdim=True)

        return grad_input.contiguous(), grad_bias, None, None


softmax = SoftmaxAffineFunction.apply
mask_softmax = FusedMaskSoftmaxFunction.apply
mask_bias_softmax = FusedMaskBiasSoftmaxFunction.apply
