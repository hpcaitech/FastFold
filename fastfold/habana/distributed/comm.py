from typing import Tuple

import torch
import torch.distributed as dist
from torch import Tensor

from .core import (ensure_divisibility, get_tensor_model_parallel_group,
                   get_tensor_model_parallel_rank,
                   get_tensor_model_parallel_world_size)


def divide(numerator, denominator):
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def _reduce(tensor: Tensor) -> Tensor:
    if dist.get_world_size() == 1:
        return tensor

    dist.all_reduce(tensor,
                    op=dist.ReduceOp.SUM,
                    group=get_tensor_model_parallel_group(),
                    async_op=False)

    return tensor


def _split(tensor: Tensor, dim: int = -1) -> Tensor:
    if get_tensor_model_parallel_world_size() == 1:
        return tensor

    split_size = divide(tensor.shape[dim], get_tensor_model_parallel_world_size())
    tensor_list = torch.split(tensor, split_size, dim=dim)

    output = tensor_list[get_tensor_model_parallel_rank()].contiguous()

    return output


def _gather(tensor: Tensor, dim: int = -1) -> Tensor:
    if get_tensor_model_parallel_world_size() == 1:
        return tensor

    if dim == 1 and list(tensor.shape)[0] == 1:
        output_shape = list(tensor.shape)
        output_shape[1] *= get_tensor_model_parallel_world_size()
        output = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
        tensor_list = output.chunk(get_tensor_model_parallel_world_size(), dim=1)
        dist.all_gather(list(tensor_list),
                        tensor,
                        group=get_tensor_model_parallel_group(),
                        async_op=False)
    else:
        tensor_list = [
            torch.empty_like(tensor) for _ in range(get_tensor_model_parallel_world_size())
        ]
        dist.all_gather(tensor_list,
                        tensor,
                        group=get_tensor_model_parallel_group(),
                        async_op=False)
        output = torch.cat(tensor_list, dim=dim)

    return output


def copy(input: Tensor) -> Tensor:
    if torch.is_grad_enabled() and input.requires_grad:
        input = Copy.apply(input)
    return input


class Copy(torch.autograd.Function):

    @staticmethod
    def forward(ctx: "Copy", input: Tensor) -> Tensor:
        return input

    @staticmethod
    def backward(ctx: "Copy", grad_output: Tensor) -> Tensor:
        return _reduce(grad_output)


def scatter(input: Tensor, dim: int = -1) -> Tensor:
    if torch.is_grad_enabled() and input.requires_grad:
        input = Scatter.apply(input, dim)
    else:
        input = _split(input, dim=dim)
    return input


class Scatter(torch.autograd.Function):

    @staticmethod
    def forward(ctx: "Scatter", input: Tensor, dim: int = -1) -> Tensor:
        ctx.save_for_backward(torch.tensor([dim]))
        return _split(input, dim=dim)

    @staticmethod
    def backward(ctx: "Scatter", grad_output: Tensor) -> Tuple[Tensor]:
        dim, = ctx.saved_tensors
        return _gather(grad_output, dim=int(dim)), None


def reduce(input: Tensor) -> Tensor:
    if torch.is_grad_enabled() and input.requires_grad:
        input = Reduce.apply(input)
    else:
        input = _reduce(input)
    return input


class Reduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx: "Reduce", input: Tensor) -> Tensor:
        return _reduce(input)

    @staticmethod
    def backward(ctx: "Reduce", grad_output: Tensor) -> Tensor:
        return grad_output


def gather(input: Tensor, dim: int = -1) -> Tensor:
    if torch.is_grad_enabled() and input.requires_grad:
        input = Gather.apply(input, dim)
    else:
        input = _gather(input, dim=dim)
    return input


class Gather(torch.autograd.Function):

    @staticmethod
    def forward(ctx: "Gather", input: Tensor, dim: int = -1) -> Tensor:
        ctx.save_for_backward(torch.tensor([dim]))
        return _gather(input, dim=dim)

    @staticmethod
    def backward(ctx: "Gather", grad_output: Tensor) -> Tuple[Tensor]:
        dim, = ctx.saved_tensors
        return _split(grad_output, dim=int(dim)), None


def _all_to_all(tensor: Tensor, in_dim: int = -1, out_dim: int = -1) -> Tensor:
    if dist.get_world_size() == 1:
        return tensor

    tensor = tensor.transpose(in_dim, 0).contiguous()

    output = torch.empty_like(tensor)
    dist.all_to_all_single(output, tensor, group=get_tensor_model_parallel_group())

    output = output.transpose(in_dim, 0).contiguous()

    tensor_list = output.chunk(get_tensor_model_parallel_world_size(), dim=in_dim)

    return torch.cat(tensor_list, dim=out_dim)


def col_to_row(input_: Tensor) -> Tensor:
    if torch.is_grad_enabled() and input_.requires_grad:
        input_ = All_to_All.apply(input_, 1, 2)
    else:
        input_ = _all_to_all(input_, in_dim=1, out_dim=2)
    return input_


def row_to_col(input_: Tensor) -> Tensor:
    if torch.is_grad_enabled() and input_.requires_grad:
        input_ = All_to_All.apply(input_, 2, 1)
    else:
        input_ = _all_to_all(input_, in_dim=2, out_dim=1)
    return input_


class All_to_All(torch.autograd.Function):

    @staticmethod
    def forward(ctx: "All_to_All", input_: Tensor, in_dim: int = -1, out_dim: int = -1) -> Tensor:
        ctx.save_for_backward(torch.tensor([in_dim, out_dim]))
        return _all_to_all(input_, in_dim=in_dim, out_dim=out_dim)

    @staticmethod
    def backward(ctx: "All_to_All", grad_output: Tensor) -> Tuple[Tensor]:
        saved_tensors = ctx.saved_tensors[0]
        return _all_to_all(grad_output, in_dim=int(saved_tensors[1]),
                           out_dim=int(saved_tensors[0])), None, None
