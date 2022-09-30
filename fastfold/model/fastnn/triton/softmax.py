from functools import reduce
from operator import mul

import torch

import triton
import triton.language as tl


@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols,
                   BLOCK_SIZE: tl.constexpr):
    # The rows of the softmax are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=float("-inf")).to(tl.float32)
    # Substract maximum for numerical stability
    row_minus_max = row - tl.max(row, axis=0)
    # Note that exponentials in Triton are fast but approximate (i.e., think __expf in CUDA)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)


@triton.jit
def softmax_kernel_two_rows(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols,
                            BLOCK_SIZE: tl.constexpr):
    # The rows of the softmax are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + 2 * row_idx * input_row_stride
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf')).to(tl.float32)
    # Substract maximum for numerical stability
    row_minus_max = row - tl.max(row, axis=0)
    # Note that exponentials in Triton are fast but approximate (i.e., think __expf in CUDA)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + 2 * row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)

    row_2 = tl.load(input_ptrs + n_cols, mask=col_offsets < n_cols,
                    other=-float('inf')).to(tl.float32)
    row_minus_max_2 = row_2 - tl.max(row_2, axis=0)
    numerator_2 = tl.exp(row_minus_max_2)
    denominator_2 = tl.sum(numerator_2, axis=0)
    softmax_output_2 = numerator_2 / denominator_2
    output_ptrs_2 = output_row_start_ptr + n_cols + col_offsets
    tl.store(output_ptrs_2, softmax_output_2, mask=col_offsets < n_cols)


@triton.jit
def softmax_grad_kernel(d_output_ptr, output_ptr, d_input_ptr, d_output_row_stride,
                        output_row_stride, d_input_row_stride, n_cols, BLOCK_SIZE: tl.constexpr,
                        is_bf16: tl.constexpr):

    row_idx = tl.program_id(0)

    output_start_ptr = output_ptr + row_idx * output_row_stride
    d_output_start_ptr = d_output_ptr + row_idx * d_output_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    output_ptrs = output_start_ptr + col_offsets
    d_output_ptrs = d_output_start_ptr + col_offsets

    output_row = tl.load(output_ptrs, mask=col_offsets < n_cols, other=float("-inf"))
    d_output_row = tl.load(d_output_ptrs, mask=col_offsets < n_cols, other=float("-inf"))

    if is_bf16:
        output_row = output_row.to(tl.float32)
        d_output_row = d_output_row.to(tl.float32)

    row_sum = tl.sum(output_row * d_output_row, axis=0)
    d_softmax_output = (d_output_row - row_sum) * output_row

    d_input_row_start_ptr = d_input_ptr + row_idx * d_input_row_stride
    d_input_ptrs = d_input_row_start_ptr + col_offsets

    tl.store(d_input_ptrs, d_softmax_output, mask=col_offsets < n_cols)


@triton.jit
def softmax_grad_kernel_two_rows(d_output_ptr, output_ptr, d_input_ptr, d_output_row_stride,
                                 output_row_stride, d_input_row_stride, n_cols,
                                 BLOCK_SIZE: tl.constexpr, is_bf16: tl.constexpr):

    row_idx = tl.program_id(0)

    output_start_ptr = output_ptr + 2 * row_idx * output_row_stride
    d_output_start_ptr = d_output_ptr + 2 * row_idx * d_output_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    output_ptrs = output_start_ptr + col_offsets
    d_output_ptrs = d_output_start_ptr + col_offsets

    output_row = tl.load(output_ptrs, mask=col_offsets < n_cols, other=float("-inf"))
    d_output_row = tl.load(d_output_ptrs, mask=col_offsets < n_cols, other=float("-inf"))

    if is_bf16:
        output_row = output_row.to(tl.float32)
        d_output_row = d_output_row.to(tl.float32)

    row_sum = tl.sum(output_row * d_output_row, axis=0)
    d_softmax_output = (d_output_row - row_sum) * output_row

    d_input_row_start_ptr = d_input_ptr + 2 * row_idx * d_input_row_stride
    d_input_ptrs = d_input_row_start_ptr + col_offsets

    tl.store(d_input_ptrs, d_softmax_output, mask=col_offsets < n_cols)

    output_row_2 = tl.load(output_ptrs + n_cols, mask=col_offsets < n_cols, other=float("-inf"))
    d_output_row_2 = tl.load(d_output_ptrs + n_cols, mask=col_offsets < n_cols, other=float("-inf"))

    if is_bf16:
        output_row_2 = output_row_2.to(tl.float32)
        d_output_row_2 = d_output_row_2.to(tl.float32)

    row_sum_2 = tl.sum(output_row_2 * d_output_row_2, axis=0)
    d_softmax_output_2 = (d_output_row_2 - row_sum_2) * output_row_2

    d_input_ptrs_2 = d_input_row_start_ptr + n_cols + col_offsets

    tl.store(d_input_ptrs_2, d_softmax_output_2, mask=col_offsets < n_cols)


def _softmax_kernel_wrapper(x):
    n_rows, n_cols = x.shape

    y = torch.empty_like(x)

    num_warps = 1
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    if BLOCK_SIZE >= 1024:
        num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    if n_cols < 256 and n_rows % 2 == 0:
        softmax_kernel_two_rows[(n_rows // 2,)](
            y,
            x,
            x.stride(-2),
            y.stride(-2),
            n_cols,
            num_warps=num_warps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        softmax_kernel[(n_rows,)](
            y,
            x,
            x.stride(-2),
            y.stride(-2),
            n_cols,
            num_warps=num_warps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    return y


def _softmax_grad_kernel_wrapper(grad_output, output, n_rows, n_cols):
    grad_input = torch.empty_like(grad_output)

    num_warps = 1
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    if BLOCK_SIZE >= 1024:
        num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    is_bf16 = (output.dtype == torch.bfloat16)

    if n_cols <= 128 and n_rows % 2 == 0:
        softmax_grad_kernel_two_rows[(n_rows // 2,)](
            grad_output,
            output,
            grad_input,
            grad_output.stride(-2),
            output.stride(-2),
            grad_output.stride(-2),
            n_cols,
            num_warps=num_warps,
            BLOCK_SIZE=BLOCK_SIZE,
            is_bf16=is_bf16,
        )
    else:
        softmax_grad_kernel[(n_rows,)](
            grad_output,
            output,
            grad_input,
            grad_output.stride(-2),
            output.stride(-2),
            grad_output.stride(-2),
            n_cols,
            num_warps=num_warps,
            BLOCK_SIZE=BLOCK_SIZE,
            is_bf16=is_bf16,
        )

    return grad_input


@triton.jit
def softmax_mask_kernel(output_ptr, input_ptr, mask_ptr, input_row_stride, output_row_stride,
                        n_cols, n_heads, BLOCK_SIZE: tl.constexpr):
    # The rows of the softmax are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    mask_start_ptr = mask_ptr + (row_idx // (n_heads * n_cols)) * n_cols
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask_ptrs = mask_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=float("-inf")).to(tl.float32)
    mask = tl.load(mask_ptrs, mask=col_offsets < n_cols, other=float("-inf")).to(tl.float32)
    row = tl.where(mask == 0, float("-1e10"), row)
    # Substract maximum for numerical stability
    row_minus_max = row - tl.max(row, axis=0)
    # Note that exponentials in Triton are fast but approximate (i.e., think __expf in CUDA)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)


@triton.jit
def softmax_mask_kernel_two_rows(output_ptr, input_ptr, mask_ptr, input_row_stride,
                                 output_row_stride, n_cols, n_heads, BLOCK_SIZE: tl.constexpr):
    # The rows of the softmax are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + 2 * row_idx * input_row_stride
    mask_start_ptr = mask_ptr + ((2 * row_idx) // (n_heads * n_cols)) * n_cols
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask_ptrs = mask_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf')).to(tl.float32)
    mask = tl.load(mask_ptrs, mask=col_offsets < n_cols, other=float("-inf")).to(tl.float32)
    row = tl.where(mask == 0, float("-1e10"), row)
    # Substract maximum for numerical stability
    row_minus_max = row - tl.max(row, axis=0)
    # Note that exponentials in Triton are fast but approximate (i.e., think __expf in CUDA)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + 2 * row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)

    row_2 = tl.load(input_ptrs + n_cols, mask=col_offsets < n_cols,
                    other=-float('inf')).to(tl.float32)
    mask_start_ptr = mask_ptr + ((2 * row_idx + 1) // (n_heads * n_cols)) * n_cols
    mask_ptrs = mask_start_ptr + col_offsets
    mask = tl.load(mask_ptrs, mask=col_offsets < n_cols, other=float("-inf")).to(tl.float32)
    row_2 = tl.where(mask == 0, float("-1e10"), row)
    row_minus_max_2 = row_2 - tl.max(row_2, axis=0)
    numerator_2 = tl.exp(row_minus_max_2)
    denominator_2 = tl.sum(numerator_2, axis=0)
    softmax_output_2 = numerator_2 / denominator_2
    output_ptrs_2 = output_row_start_ptr + n_cols + col_offsets
    tl.store(output_ptrs_2, softmax_output_2, mask=col_offsets < n_cols)


@triton.jit
def softmax_mask_grad_kernel(d_output_ptr, output_ptr, d_input_ptr, mask_ptr, d_output_row_stride,
                             output_row_stride, d_input_row_stride, n_cols, n_heads,
                             BLOCK_SIZE: tl.constexpr, is_bf16: tl.constexpr):

    row_idx = tl.program_id(0)

    output_start_ptr = output_ptr + row_idx * output_row_stride
    d_output_start_ptr = d_output_ptr + row_idx * d_output_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    output_ptrs = output_start_ptr + col_offsets
    d_output_ptrs = d_output_start_ptr + col_offsets

    output_row = tl.load(output_ptrs, mask=col_offsets < n_cols, other=float("-inf"))
    d_output_row = tl.load(d_output_ptrs, mask=col_offsets < n_cols, other=float("-inf"))

    if is_bf16:
        output_row = output_row.to(tl.float32)
        d_output_row = d_output_row.to(tl.float32)

    row_sum = tl.sum(output_row * d_output_row, axis=0)
    d_softmax_output = (d_output_row - row_sum) * output_row

    d_input_row_start_ptr = d_input_ptr + row_idx * d_input_row_stride
    d_input_ptrs = d_input_row_start_ptr + col_offsets

    mask_start_ptr = mask_ptr + (row_idx // (n_heads * n_cols)) * n_cols
    mask_ptrs = mask_start_ptr + col_offsets
    mask = tl.load(mask_ptrs, mask=col_offsets < n_cols, other=float("-inf")).to(tl.float32)
    d_softmax_output = tl.where(mask == 0, float(0), d_softmax_output)

    tl.store(d_input_ptrs, d_softmax_output, mask=col_offsets < n_cols)


@triton.jit
def softmax_mask_grad_kernel_two_rows(d_output_ptr, output_ptr, d_input_ptr, mask_ptr,
                                      d_output_row_stride, output_row_stride, d_input_row_stride,
                                      n_cols, n_heads, BLOCK_SIZE: tl.constexpr,
                                      is_bf16: tl.constexpr):

    row_idx = tl.program_id(0)

    output_start_ptr = output_ptr + 2 * row_idx * output_row_stride
    d_output_start_ptr = d_output_ptr + 2 * row_idx * d_output_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    output_ptrs = output_start_ptr + col_offsets
    d_output_ptrs = d_output_start_ptr + col_offsets

    output_row = tl.load(output_ptrs, mask=col_offsets < n_cols, other=float("-inf"))
    d_output_row = tl.load(d_output_ptrs, mask=col_offsets < n_cols, other=float("-inf"))

    if is_bf16:
        output_row = output_row.to(tl.float32)
        d_output_row = d_output_row.to(tl.float32)

    row_sum = tl.sum(output_row * d_output_row, axis=0)
    d_softmax_output = (d_output_row - row_sum) * output_row

    d_input_row_start_ptr = d_input_ptr + 2 * row_idx * d_input_row_stride
    d_input_ptrs = d_input_row_start_ptr + col_offsets

    mask_start_ptr = mask_ptr + (2 * row_idx // (n_heads * n_cols)) * n_cols
    mask_ptrs = mask_start_ptr + col_offsets
    mask = tl.load(mask_ptrs, mask=col_offsets < n_cols, other=float("-inf")).to(tl.float32)
    d_softmax_output = tl.where(mask == 0, float(0), d_softmax_output)

    tl.store(d_input_ptrs, d_softmax_output, mask=col_offsets < n_cols)

    output_row_2 = tl.load(output_ptrs + n_cols, mask=col_offsets < n_cols, other=float("-inf"))
    d_output_row_2 = tl.load(d_output_ptrs + n_cols, mask=col_offsets < n_cols, other=float("-inf"))

    if is_bf16:
        output_row_2 = output_row_2.to(tl.float32)
        d_output_row_2 = d_output_row_2.to(tl.float32)

    row_sum_2 = tl.sum(output_row_2 * d_output_row_2, axis=0)
    d_softmax_output_2 = (d_output_row_2 - row_sum_2) * output_row_2

    d_input_ptrs_2 = d_input_row_start_ptr + n_cols + col_offsets

    mask_start_ptr = mask_ptr + ((2 * row_idx + 1) // (n_heads * n_cols)) * n_cols
    mask_ptrs = mask_start_ptr + col_offsets
    mask = tl.load(mask_ptrs, mask=col_offsets < n_cols, other=float("-inf")).to(tl.float32)
    d_softmax_output_2 = tl.where(mask == 0, float(0), d_softmax_output_2)
    tl.store(d_input_ptrs_2, d_softmax_output_2, mask=col_offsets < n_cols)


def _softmax_mask_kernel_wrapper(x, mask, n_rows, n_cols):
    y = torch.empty_like(x)
    n_heads = x.shape[2]

    num_warps = 1
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    if BLOCK_SIZE >= 1024:
        num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    if n_cols < 256 and n_rows % 2 == 0:
        softmax_mask_kernel_two_rows[(n_rows // 2,)](
            y,
            x,
            mask,
            x.stride(-2),
            y.stride(-2),
            n_cols,
            n_heads,
            num_warps=num_warps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        softmax_mask_kernel[(n_rows,)](
            y,
            x,
            mask,
            x.stride(-2),
            y.stride(-2),
            n_cols,
            n_heads,
            num_warps=num_warps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    return y


def _softmax_mask_grad_kernel_wrapper(grad_output, output, mask, n_rows, n_cols):
    grad_input = torch.empty_like(grad_output)
    n_heads = output.shape[2]

    num_warps = 1
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    if BLOCK_SIZE >= 1024:
        num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    is_bf16 = (output.dtype == torch.bfloat16)

    if n_cols <= 128 and n_rows % 2 == 0:
        softmax_mask_grad_kernel_two_rows[(n_rows // 2,)](
            grad_output,
            output,
            grad_input,
            mask,
            grad_output.stride(-2),
            output.stride(-2),
            grad_output.stride(-2),
            n_cols,
            n_heads,
            num_warps=num_warps,
            BLOCK_SIZE=BLOCK_SIZE,
            is_bf16=is_bf16,
        )
    else:
        softmax_mask_grad_kernel[(n_rows,)](
            grad_output,
            output,
            grad_input,
            mask,
            grad_output.stride(-2),
            output.stride(-2),
            grad_output.stride(-2),
            n_cols,
            n_heads,
            num_warps=num_warps,
            BLOCK_SIZE=BLOCK_SIZE,
            is_bf16=is_bf16,
        )
    return grad_input


@triton.jit
def softmax_mask_bias_kernel(output_ptr, input_ptr, mask_ptr, bias_ptr, input_row_stride,
                             output_row_stride, n_cols, n_heads, BLOCK_SIZE: tl.constexpr):
    # The rows of the softmax are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    mask_start_ptr = mask_ptr + (row_idx // (n_heads * n_cols)) * n_cols
    bias_start_ptr = bias_ptr + (row_idx % (n_heads * n_cols)) * n_cols
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask_ptrs = mask_start_ptr + col_offsets
    bias_ptrs = bias_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=float("-inf")).to(tl.float32)
    bias = tl.load(bias_ptrs, mask=col_offsets < n_cols, other=float("-inf")).to(tl.float32)
    row = row + bias
    mask = tl.load(mask_ptrs, mask=col_offsets < n_cols, other=float("-inf")).to(tl.float32)
    row = tl.where(mask == 0, float("-1e10"), row)
    # Substract maximum for numerical stability
    row_minus_max = row - tl.max(row, axis=0)
    # Note that exponentials in Triton are fast but approximate (i.e., think __expf in CUDA)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)


@triton.jit
def softmax_mask_bias_kernel_two_rows(output_ptr, input_ptr, mask_ptr, bias_ptr, input_row_stride,
                                      output_row_stride, n_cols, n_heads, BLOCK_SIZE: tl.constexpr):
    # The rows of the softmax are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + 2 * row_idx * input_row_stride
    mask_start_ptr = mask_ptr + ((2 * row_idx) // (n_heads * n_cols)) * n_cols
    bias_start_ptr = bias_ptr + ((2 * row_idx) % (n_heads * n_cols)) * n_cols
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask_ptrs = mask_start_ptr + col_offsets
    bias_ptrs = bias_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=float("-inf")).to(tl.float32)
    bias = tl.load(bias_ptrs, mask=col_offsets < n_cols, other=float("-inf")).to(tl.float32)
    mask = tl.load(mask_ptrs, mask=col_offsets < n_cols, other=float("-inf")).to(tl.float32)
    row = row + bias
    row = tl.where(mask == 0, float("-1e10"), row)
    # Substract maximum for numerical stability
    row_minus_max = row - tl.max(row, axis=0)
    # Note that exponentials in Triton are fast but approximate (i.e., think __expf in CUDA)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + 2 * row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)

    bias_start_ptr = bias_ptr + ((2 * row_idx + 1) % (n_heads * n_cols)) * n_cols
    mask_start_ptr = mask_ptr + ((2 * row_idx + 1) // (n_heads * n_cols)) * n_cols
    bias_ptrs = bias_start_ptr + col_offsets
    mask_ptrs = mask_start_ptr + col_offsets
    row_2 = tl.load(input_ptrs + n_cols, mask=col_offsets < n_cols,
                    other=-float('inf')).to(tl.float32)
    bias = tl.load(bias_ptrs, mask=col_offsets < n_cols, other=float("-inf")).to(tl.float32)
    mask = tl.load(mask_ptrs, mask=col_offsets < n_cols, other=float("-inf")).to(tl.float32)
    row = row + bias
    row_2 = tl.where(mask == 0, float("-1e10"), row)
    row_minus_max_2 = row_2 - tl.max(row_2, axis=0)
    numerator_2 = tl.exp(row_minus_max_2)
    denominator_2 = tl.sum(numerator_2, axis=0)
    softmax_output_2 = numerator_2 / denominator_2
    output_ptrs_2 = output_row_start_ptr + n_cols + col_offsets
    tl.store(output_ptrs_2, softmax_output_2, mask=col_offsets < n_cols)


def _softmax_mask_bias_kernel_wrapper(x, mask, bias, n_rows, n_cols):
    y = torch.empty_like(x)
    n_heads = x.shape[2]

    num_warps = 1
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    if BLOCK_SIZE >= 1024:
        num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    if n_cols < 256 and n_rows % 2 == 0:
        softmax_mask_bias_kernel_two_rows[(n_rows // 2,)](
            y,
            x,
            mask,
            bias,
            x.stride(-2),
            y.stride(-2),
            n_cols,
            n_heads,
            num_warps=num_warps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        softmax_mask_bias_kernel[(n_rows,)](
            y,
            x,
            mask,
            bias,
            x.stride(-2),
            y.stride(-2),
            n_cols,
            n_heads,
            num_warps=num_warps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    return y


class SoftmaxTritonFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        input_ = input.contiguous()
        ctx.cols = input_.shape[-1]
        ctx.rows = reduce(mul, input.shape[:-1])
        output = _softmax_kernel_wrapper(input_)
        ctx.save_for_backward(output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        output = ctx.saved_tensors[0]
        return _softmax_grad_kernel_wrapper(grad_output.contiguous(), output, ctx.rows, ctx.cols)


class FusedMaskSoftmaxTritonFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, mask):
        input_, mask_ = input.contiguous(), mask.contiguous()
        ctx.cols = input_.shape[-1]
        ctx.rows = reduce(mul, input.shape[:-1])
        output = _softmax_mask_kernel_wrapper(input_, mask_, ctx.rows, ctx.cols)
        ctx.save_for_backward(output, mask_)
        return output

    @staticmethod
    def backward(ctx, grad_output):

        output, mask_ = ctx.saved_tensors
        grad_input = _softmax_mask_grad_kernel_wrapper(grad_output.contiguous(), output, mask_,
                                                       ctx.rows, ctx.cols)

        return grad_input, None


class FusedMaskBiasSoftmaxTritonFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, mask, bias):
        input_, mask_, bias_ = input.contiguous(), mask.contiguous(), bias.contiguous()
        ctx.cols = input_.shape[-1]
        ctx.rows = reduce(mul, input.shape[:-1])
        output = _softmax_mask_bias_kernel_wrapper(input_, mask_, bias_, ctx.rows, ctx.cols)
        ctx.save_for_backward(output, mask_)
        return output

    @staticmethod
    def backward(ctx, grad_output):

        output, mask_ = ctx.saved_tensors
        grad_input = _softmax_mask_grad_kernel_wrapper(grad_output.contiguous(), output, mask_,
                                                       ctx.rows, ctx.cols)
        grad_bias = torch.sum(grad_input, dim=1, keepdim=True)

        return grad_input, None, grad_bias


softmax = SoftmaxTritonFunc.apply
mask_softmax = FusedMaskSoftmaxTritonFunc.apply
mask_bias_softmax = FusedMaskBiasSoftmaxTritonFunc.apply
