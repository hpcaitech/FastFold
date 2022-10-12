import torch
from fastfold.model.fastnn.ops import Linear as FastLinear
from fastfold.model.nn.primitives import Linear


def test_linear():    
    c_in = 3
    c_out = 4
    seq = 5
    
    fast_linear = FastLinear(c_in, c_out).cuda()
    linear = Linear(c_in, c_out).cuda()
    
    fast_linear.weight = linear.weight
    fast_linear.bias = linear.bias

    x = torch.randn((seq, c_in)).cuda()

    out1 = fast_linear(x)
    out2 = linear(x)
    assert torch.allclose(out1, out2, atol=1e-8)


if __name__ == "__main__":
    test_linear()
