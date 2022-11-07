import math

import pytest

import torch
from einops import rearrange

TEST_TRITON = False
try:
    from fastfold.model.fastnn.kernel import fused_attention_core
except:
    print("Skip triton attention test!")
    TEST_TRITON = False


def torch_core_attention(q, k, v, mask, bias):

    scaling = 1. / math.sqrt(q.size(-1))
    q = q * scaling

    logits = torch.matmul(q.float(), k.float().transpose(-1, -2))
    logits += bias.float()
    logits += (1e20 * (mask - 1))[..., :, None, None, :]

    weights = torch.nn.functional.softmax(logits.float(), -1).to(dtype=q.dtype)

    weighted_avg = torch.matmul(weights, v)

    weighted_avg = rearrange(weighted_avg, 'b1 b2 h n d -> b1 b2 n (h d)')

    return weighted_avg

@pytest.mark.skipif(TEST_TRITON == False, reason="triton is not available")
def test_fused_attention_core():
    if TEST_TRITON:
        batch_, chunk_, head_, d_head = 1, 8, 4, 32
        test_seq_ = [32, 256, 370, 500, 512, 700, 1024, 1600]
        test_dtype = [torch.float16, torch.bfloat16]
        test_device = torch.device("cuda")

        tolerance_eps = {torch.float16: 1e-4, torch.bfloat16: 1e-4}

        for seq_ in test_seq_:
            for dtype in test_dtype:
                q = torch.empty((batch_, chunk_, head_, seq_, d_head), dtype=dtype,
                                device="cuda").normal_(mean=0, std=.5).requires_grad_()
                k = torch.empty((batch_, chunk_, head_, seq_, d_head), dtype=dtype,
                                device="cuda").normal_(mean=0, std=.5).requires_grad_()
                v = torch.empty((batch_, chunk_, head_, seq_, d_head), dtype=dtype,
                                device="cuda").normal_(mean=0, std=.5).requires_grad_()

                mask = torch.empty(
                    (batch_, chunk_, seq_), device="cuda").normal_(mean=0, std=.5) > 0
                mask = mask.to(device=test_device, dtype=dtype).requires_grad_(False)

                bias = torch.randn(batch_, head_, seq_, seq_).to(device=test_device,
                                                                 dtype=dtype).requires_grad_(True)

                ref_out = torch_core_attention(q, k, v, mask, bias)
                tri_out = fused_attention_core(q, k, v, mask, bias)
                # compare
                torch.allclose(ref_out, tri_out, atol=tolerance_eps[dtype])


if __name__ == "__main__":
    test_fused_attention_core()
