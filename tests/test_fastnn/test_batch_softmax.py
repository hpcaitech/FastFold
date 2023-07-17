# Testing the triton edition softmax only
# For triton softmax support an additional batch

import torch
import pytest
from fastfold.model.fastnn.kernel import fused_softmax


TEST_TRITON = True
try:
    import triton
except:
    print("Skip triton attention test!")
    TEST_TRITON = False

def _test_softmax_core():

    batch, batch_, chunk_, head_ = 3, 1, 8, 4
    test_seq_ = [31, 32, 128, 129, 256, 259, 512, 700, 1024]
    test_dtype = [torch.float32, torch.float16, torch.bfloat16]
    test_device = torch.device("cuda")

    tolerance_eps = {torch.float32: 1e-6, torch.float16: 2e-4, torch.bfloat16: 1e-3}

    for seq_ in test_seq_:
        for dtype in test_dtype:
            sample_input = torch.rand(batch, batch_, chunk_, head_, seq_,
                                      seq_).to(device=test_device, dtype=dtype).requires_grad_(True)
            sample_mask = torch.cuda.FloatTensor(batch, batch_, chunk_, seq_).uniform_() > 0
            sample_mask = sample_mask.to(device=test_device, dtype=dtype).requires_grad_(False)
            sample_bias = torch.rand(batch, batch_, 1, head_, seq_,
                                      seq_).to(device=test_device, dtype=dtype).requires_grad_(True)

            sample_input_fastnn = torch.clone(sample_input.detach()).requires_grad_(True)
            sample_mask_fastnn = torch.clone(sample_mask.detach()).requires_grad_(False)
            sample_bias_fastnn = torch.clone(sample_bias.detach()).requires_grad_(True)

            # Forward
            sample_mask_torch = 1e9 * (sample_mask - 1)[:, :, :, None, None, :]
            torch_out = torch.nn.functional.softmax(sample_input + sample_mask_torch + sample_bias,
                                                    dim=-1)

            fastnn_out = fused_softmax(sample_input_fastnn, sample_mask_fastnn, sample_bias_fastnn)
            # print(sample_bias_fastnn)
            # print(fastnn_out)
            fwd_fastnn_error = torch.max(torch.abs(torch_out - fastnn_out)).cpu().item()
            assert fwd_fastnn_error < tolerance_eps[
                dtype], f"fastnn fwd kernel error when {seq_} {dtype}"

            # Backward
            out_grad = torch.rand_like(torch_out).requires_grad_(False)
            torch_out.backward(out_grad)
            fastnn_out.backward(out_grad)

            grad_input_error = torch.max(torch.abs(sample_input.grad -
                                                   sample_input_fastnn.grad)).cpu().item()
            assert grad_input_error < tolerance_eps[
                dtype], f"fastnn bwd kernel error when {seq_} {dtype}"

            grad_bias_error = torch.max(torch.abs(sample_bias.grad -
                                                  sample_bias_fastnn.grad)).cpu().item()
            assert grad_bias_error < tolerance_eps[
                dtype], f"fastnn bwd kernel error when {seq_} {dtype}"


@pytest.mark.skipif(TEST_TRITON == False, reason="triton is not available")
def test_softmax():
    _test_softmax_core()

if __name__ == "__main__":
    test_softmax()
