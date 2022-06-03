import torch
from fastfold.model.fastnn.kernel import softmax


def test_softmax():

    # [batch, dim]
    test_shape = [[64, 64], [64, 128], [64, 129], [64, 1024]]
    test_dtype = [torch.float32, torch.float16, torch.bfloat16]
    test_device = torch.device("cuda")

    tolerance_eps = {torch.float32: 10e-5, torch.float16: 10e-2, torch.bfloat16: 10e-2}

    for shape in test_shape:
        for dtype in test_dtype:
            sample_input = torch.rand(shape).to(device=test_device,
                                                dtype=dtype).requires_grad_(True)

            sample_input_fastnn = torch.clone(sample_input.detach()).requires_grad_(True)

            # Forward
            torch_out = torch.nn.functional.softmax(sample_input, dim=-1)
            fastnn_out = softmax(sample_input_fastnn)
            forward_error = torch.max(torch.abs(torch_out - fastnn_out)).cpu().item()
            assert forward_error < tolerance_eps[dtype], f"Error when {shape} {dtype}"

            # Backward
            out_grad = torch.rand_like(torch_out).requires_grad_(False)
            torch_out.backward(out_grad)
            fastnn_out.backward(out_grad)

            backward_error = torch.max(torch.abs(sample_input.grad -
                                                 sample_input_fastnn.grad)).cpu().item()
            assert backward_error < tolerance_eps[dtype], f"Error when {shape} {dtype}"
