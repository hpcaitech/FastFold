import torch
from fastfold.model.fastnn.kernel import LayerNorm as FastLayerNorm
from fastfold.model.fastnn.kernel.layer_norm import FusedLayerNormAffineFunction

triton = True
try:
    from fastfold.model.fastnn.kernel.layer_norm import LayerNormTritonFunc
except:
    print("Skip triton layernorm test!")
    triton = False


def test_layernorm():

    # [batch, dim]
    test_shape = [[64, 64], [64, 128], [64, 129], [64, 1024]]
    test_dtype = [torch.float32, torch.float16, torch.bfloat16]
    test_device = torch.device("cuda")

    tolerance_eps = {torch.float32: 10e-5, torch.float16: 10e-2, torch.bfloat16: 10e-2}

    for shape in test_shape:
        for dtype in test_dtype:
            sample_input = torch.rand(shape).to(device=test_device,
                                                dtype=dtype).requires_grad_(False)

            dim_ = sample_input.size()[-1]
            torch_module = torch.nn.LayerNorm(normalized_shape=dim_).to(device=test_device,
                                                                        dtype=dtype)
            fastnn_cuda_module = FastLayerNorm(normalized_shape=dim_).to(device=test_device, dtype=dtype)
            if triton:
                fastnn_triton_module = FastLayerNorm(normalized_shape=dim_).to(device=test_device, dtype=dtype)

            # Forward
            torch_out = torch_module(sample_input)
            
            fastnn_cuda_out = FusedLayerNormAffineFunction.apply(sample_input, fastnn_cuda_module.weight, fastnn_cuda_module.bias, 
                                                                 fastnn_cuda_module.normalized_shape, fastnn_cuda_module.eps)
            forward_error = torch.max(torch.abs(torch_out - fastnn_cuda_out)).cpu().item()
            assert forward_error < tolerance_eps[dtype], f"Error when {shape} {dtype}"
            
            if triton:
                fastnn_triton_out = LayerNormTritonFunc.apply(sample_input, fastnn_triton_module.normalized_shape, fastnn_triton_module.weight, 
                                                            fastnn_triton_module.bias, fastnn_triton_module.eps)
                forward_error = torch.max(torch.abs(torch_out - fastnn_triton_out)).cpu().item()
                assert forward_error < tolerance_eps[dtype], f"Error when {shape} {dtype}"

            # Backward
            out_grad = torch.rand_like(torch_out).requires_grad_(False)
            torch_out.backward(out_grad)
            fastnn_cuda_out.backward(out_grad)

            backward_weight_error = torch.max(
                torch.abs(torch_module.weight.grad - fastnn_cuda_module.weight.grad)).cpu().item()
            assert backward_weight_error < tolerance_eps[dtype], f"Error when {shape} {dtype}"
            backward_bias_error = torch.max(
                torch.abs(torch_module.bias.grad - fastnn_cuda_module.bias.grad)).cpu().item()
            assert backward_bias_error < tolerance_eps[dtype], f"Error when {shape} {dtype}"

            if triton:
                fastnn_triton_out.backward(out_grad)
                backward_weight_error = torch.max(
                    torch.abs(torch_module.weight.grad - fastnn_triton_module.weight.grad)).cpu().item()
                assert backward_weight_error < tolerance_eps[dtype], f"Error when {shape} {dtype}"
                backward_bias_error = torch.max(
                    torch.abs(torch_module.bias.grad - fastnn_triton_module.bias.grad)).cpu().item()
                assert backward_bias_error < tolerance_eps[dtype], f"Error when {shape} {dtype}"


if __name__ == "__main__":
    test_layernorm()
