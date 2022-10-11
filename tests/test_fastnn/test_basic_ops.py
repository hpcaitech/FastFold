import torch
from fastfold.model.fastnn.kernel import softmax
from torch import nn
from typing import Optional, Callable, List, Tuple, Sequence
from functools import partial
import importlib
import math
from typing import Optional, Callable, List, Tuple, Sequence
import numpy as np
from scipy.stats import truncnorm


class Linear(nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just
    like torch.nn.Linear.
    Implements the initializers in 1.11.4, plus some additional ones found
    in the code.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
    ):
        """
        Args:
            in_dim:
                The final dimension of inputs to the layer
            out_dim:
                The final dimension of layer outputs
            bias:
                Whether to learn an additive bias. True by default
            init:
                The initializer to use. Choose from:
                "default": LeCun fan-in truncated normal initialization
                "relu": He initialization w/ truncated normal distribution
                "glorot": Fan-average Glorot uniform initialization
                "gating": Weights=0, Bias=1
                "normal": Normal initialization with std=1/sqrt(fan_in)
                "final": Weights=0, Bias=0
                Overridden by init_fn if the latter is not None.
            init_fn:
                A custom initializer taking weight and bias as inputs.
                Overrides init if not None.
        """
        super(Linear, self).__init__(in_dim, out_dim, bias=bias)
        
        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        with torch.no_grad():
            if init_fn is not None:
                init_fn(self.weight, self.bias)
            else:
                if init == "default":
                    self.lecun_normal_init_(self.weight)
                elif init == "relu":
                    self.he_normal_init_(self.weight)
                elif init == "glorot":
                    self.glorot_uniform_init_(self.weight)
                elif init == "gating":
                    self.gating_init_(self.weight)
                    if bias:
                        self.bias.fill_(1.0)
                elif init == "normal":
                    self.normal_init_(self.weight)
                elif init == "final":
                    self.final_init_(self.weight)
                else:
                    raise ValueError("Invalid init string.")
    
    def _prod(self, nums):
        out = 1
        for n in nums:
            out = out * n
        return out

    def _calculate_fan(self, linear_weight_shape, fan="fan_in"):
        fan_out, fan_in = linear_weight_shape
        if fan == "fan_in":
            f = fan_in
        elif fan == "fan_out":
            f = fan_out
        elif fan == "fan_avg":
            f = (fan_in + fan_out) / 2
        else:
            raise ValueError("Invalid fan option")
        return f

    def trunc_normal_init_(self, weights, scale=1.0, fan="fan_in"):
        shape = weights.shape
        f = self._calculate_fan(shape, fan)
        scale = scale / max(1, f)
        a = -2
        b = 2
        std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
        size = self._prod(shape)
        samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
        samples = np.reshape(samples, shape)
        with torch.no_grad():
            weights.copy_(torch.tensor(samples, device=weights.device))

    def lecun_normal_init_(self, weights):
        self.trunc_normal_init_(weights, scale=1.0)

    def he_normal_init_(self, weights):
        self.trunc_normal_init_(weights, scale=2.0)

    def glorot_uniform_init_(self, weights):
        nn.init.xavier_uniform_(weights, gain=1)

    def final_init_(self, weights):
        with torch.no_grad():
            weights.fill_(0.0)

    def gating_init_(self, weights):
        with torch.no_grad():
            weights.fill_(0.0)

    def normal_init_(self, weights):
        torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")

    def ipa_point_weights_init_(self, weights):
        with torch.no_grad():
            softplus_inverse_1 = 0.541324854612918
            weights.fill_(softplus_inverse_1)