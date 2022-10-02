import importlib

import torch

fastfold_layer_norm_cuda = importlib.import_module("fastfold_layer_norm_cuda")


class FusedLayerNormAffineFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias, normalized_shape, eps):

        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        input_ = input.contiguous()
        weight_ = weight.contiguous()
        bias_ = bias.contiguous()
        output, mean, invvar = fastfold_layer_norm_cuda.forward_affine(
            input_, ctx.normalized_shape, weight_, bias_, ctx.eps)
        ctx.save_for_backward(input_, weight_, bias_, mean, invvar)

        return output

    @staticmethod
    def backward(ctx, grad_output):

        input_, weight_, bias_, mean, invvar = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        grad_input, grad_weight, grad_bias \
            = fastfold_layer_norm_cuda.backward_affine(
                grad_output.contiguous(), mean, invvar,
                input_, ctx.normalized_shape,
                weight_, bias_, ctx.eps)

        return grad_input, grad_weight, grad_bias, None, None
