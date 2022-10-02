import numbers
import logging

import torch
from torch.nn.parameter import Parameter

_triton_available = True
if _triton_available:
    try:
        from .triton.layer_norm import LayerNormTritonFunc

    except ImportError:
        logging.warning("Triton is not available, fallback to old kernel.")
        _triton_available = False

from .cuda_native.layer_norm import FusedLayerNormAffineFunction


class FusedLayerNorm(torch.nn.Module):

    def __init__(self, normalized_shape, eps=1e-5):
        super(FusedLayerNorm, self).__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.weight = Parameter(torch.Tensor(*normalized_shape))
        self.bias = Parameter(torch.Tensor(*normalized_shape))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, input):
        if _triton_available:
            return LayerNormTritonFunc.apply(input, self.normalized_shape, self.weight, self.bias,
                                             self.eps)
        else:
            return FusedLayerNormAffineFunction.apply(input, self.weight, self.bias,
                                                      self.normalized_shape, self.eps)
