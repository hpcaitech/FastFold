###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import torch
from fusedsoftmax import fused_softmax, fused_softmax_bias

def test_fusedsoftmax_op_function():
    print(torch.ops.custom_op.fusedsoftmax)
    print(torch.ops.custom_op.fusedsoftmax_bias)

    # print(torch.ops.custom_op.custom_relu_backward)
    input = torch.randn(1, 512, 4, 512, 512)
    mask =  torch.randn(1, 512, 1,   1, 512)
    bias =  torch.randn(1, 1, 4, 512, 512)
    dim = -1

    input_hpu = input.to('hpu')
    mask_hpu = mask.to('hpu')

    out = input + mask
    output_cpu = torch.softmax(out, dim=dim)

    output_hpu = fused_softmax(input_hpu, mask_hpu, dim)

    assert((abs(output_hpu.cpu() - output_cpu) < 1e-6).all())
    print("fused_softmax test passed")

    input_hpu = input.to('hpu')
    mask_hpu = mask.to('hpu')
    bias_hpu = bias.to('hpu')
    out = input + mask
    out += bias
    output_cpu = torch.softmax(out, dim=dim)
    
    output_hpu = fused_softmax_bias(input_hpu, mask_hpu, bias_hpu, dim);

    assert((abs(output_hpu.cpu() - output_cpu) < 1e-6).all())
    print("fused_softmax_bias test passed")

test_fusedsoftmax_op_function()


def test_fusedsoftmax_bias_op_backward_function():
    print("fused_softmax_bias_backward")
    input = torch.randn(1, 512, 4, 512, 512, requires_grad=True)
    mask =  torch.randn(1, 512, 1,   1, 512, requires_grad=False)
    bias =  torch.randn(1, 1,   4, 512, 512, requires_grad=True)
    dim = -1

    # cpu reference
    add_mask_cpu = input + mask
    add_mask_cpu += bias
    softmax_cpu = torch.softmax(add_mask_cpu, dim=dim)

    input_hpu = input.to('hpu').detach()
    input_hpu.requires_grad = True
    mask_hpu = mask.to('hpu').detach()
    mask_hpu.requires_grad = False
    bias_hpu = bias.to('hpu').detach()
    bias_hpu.requires_grad = True
    softmax_hpu = fused_softmax_bias(input_hpu, mask_hpu, bias_hpu, dim)

    assert((abs(softmax_hpu.detach().cpu() - softmax_cpu.detach()) < 1e-6).all())

    grad_cpu = torch.ones_like(softmax_cpu)
    softmax_cpu.backward(grad_cpu)
    grad_hpu = grad_cpu.to('hpu')
    softmax_hpu.backward(grad_hpu)

    input_bwd_cpu = input.grad
    input_bwd_hpu = input_hpu.grad
    assert((abs(input_bwd_hpu.detach().cpu() - input_bwd_cpu.detach()) < 1e-6).all())
    bias_bwd_cpu = bias.grad
    bias_bwd_hpu = bias_hpu.grad
    assert((abs(bias_bwd_hpu.detach().cpu() - bias_bwd_cpu.detach()) < 1e-6).all())

    print("fused_softmax_bias_backward test passed")


test_fusedsoftmax_bias_op_backward_function()

def test_fusedsoftmax_op_backward_function():
    print(torch.ops.custom_op.fusedsoftmax_backward)
    input = torch.randn(1, 512, 4, 512, 512, requires_grad=True)
    mask =  torch.randn(1, 512, 1,   1, 512, requires_grad=False)
    dim = -1

    # cpu reference
    add_mask_cpu = input + mask
    softmax_cpu = torch.softmax(add_mask_cpu, dim=dim)

    input_hpu = input.to('hpu').detach()
    input_hpu.requires_grad = True
    mask_hpu = mask.to('hpu').detach()
    mask_hpu.requires_grad = False
    softmax_hpu = fused_softmax(input_hpu, mask_hpu, dim)

    assert((abs(softmax_hpu.detach().cpu() - softmax_cpu.detach()) < 1e-6).all())

    grad_cpu = torch.ones_like(softmax_cpu)
    softmax_cpu.backward(grad_cpu)
    grad_hpu = grad_cpu.to('hpu')
    softmax_hpu.backward(grad_hpu)

    input_bwd_cpu = input.grad
    input_bwd_hpu = input_hpu.grad
    assert((abs(input_bwd_hpu.detach().cpu() - input_bwd_cpu.detach()) < 1e-6).all())

    print("fused_softmax_backward test passed")


test_fusedsoftmax_op_backward_function()
