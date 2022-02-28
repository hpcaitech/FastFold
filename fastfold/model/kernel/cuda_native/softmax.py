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
        output = fastfold_softmax_cuda.forward_affine(input_, ctx.rows, ctx.cols)
        ctx.save_for_backward(output)

        return output

    @staticmethod
    def backward(ctx, grad_output):

        output = ctx.saved_tensors[0]

        grad_input = None
        grad_input = fastfold_softmax_cuda.backward_affine(grad_output.contiguous(), output,
                                                           ctx.rows, ctx.cols)

        return grad_input


class FusedScaleMaskSoftmaxFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, mask, scale):
        input_ = input.contiguous()
        mask_ = mask.contiguous()
        ctx.cols = input_.shape[-1]
        ctx.rows = reduce(mul, input.shape[:-1])
        output = fastfold_softmax_cuda.fused_scale_mask_softmax_forward(
            input_, mask_, ctx.rows, ctx.cols, scale)
        ctx.save_for_backward(output, mask_)
        ctx.scale = scale

        return output

    @staticmethod
    def backward(ctx, grad_output):

        output, mask_ = ctx.saved_tensors

        grad_input = None
        grad_input = fastfold_softmax_cuda.fused_scale_mask_softmax_backward(
            grad_output.contiguous(), output, mask_, ctx.rows, ctx.cols, ctx.scale)

        return grad_input.contiguous(), None, None


class FusedScaleMaskBiasSoftmaxFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, mask, bias, scale):
        input_ = input.contiguous()
        mask_ = mask.contiguous()
        bias_ = bias.contiguous()
        ctx.cols = input_.shape[-1]
        ctx.rows = reduce(mul, input.shape[:-1])
        output = fastfold_softmax_cuda.fused_scale_mask_bias_softmax_forward(
            input_, mask_, bias_, ctx.rows, ctx.cols, scale)
        ctx.save_for_backward(output, mask_, bias_)
        ctx.scale = scale

        return output

    @staticmethod
    def backward(ctx, grad_output):

        output, mask_, bias_ = ctx.saved_tensors

        grad_input = None
        grad_input = fastfold_softmax_cuda.fused_scale_mask_bias_softmax_backward(
            grad_output.contiguous(), output, mask_, bias_, ctx.rows, ctx.cols, ctx.scale)

        grad_input = grad_input.contiguous()

        grad_bias = torch.sum(grad_input, dim=1, keepdim=True)

        return grad_input.contiguous(), grad_bias, None, None


softmax = SoftmaxAffineFunction.apply
scale_mask_softmax = FusedScaleMaskSoftmaxFunction.apply
scale_mask_bias_softmax = FusedScaleMaskBiasSoftmaxFunction.apply
