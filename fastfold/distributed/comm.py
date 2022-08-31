from typing import Tuple

import torch
import torch.distributed as dist
from torch import Tensor

from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc

from .core import ensure_divisibility


def divide(numerator, denominator):
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def _reduce(tensor: Tensor) -> Tensor:
    if gpc.get_world_size(ParallelMode.TENSOR) == 1:
        return tensor

    dist.all_reduce(tensor,
                    op=dist.ReduceOp.SUM,
                    group=gpc.get_group(ParallelMode.TENSOR),
                    async_op=False)

    return tensor


def _split(tensor: Tensor, dim: int = -1) -> Tensor:
    if gpc.get_world_size(ParallelMode.TENSOR) == 1:
        return tensor

    split_size = divide(tensor.shape[dim], gpc.get_world_size(ParallelMode.TENSOR))
    tensor_list = torch.split(tensor, split_size, dim=dim)

    output = tensor_list[gpc.get_local_rank(ParallelMode.TENSOR)].contiguous()

    return output


def _gather(tensor: Tensor, dim: int = -1) -> Tensor:
    if gpc.get_world_size(ParallelMode.TENSOR) == 1:
        return tensor

    if dim == 1 and list(tensor.shape)[0] == 1:
        output_shape = list(tensor.shape)
        output_shape[1] *= gpc.get_world_size(ParallelMode.TENSOR)
        output = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
        tensor_list = output.chunk(gpc.get_world_size(ParallelMode.TENSOR), dim=1)
        dist.all_gather(list(tensor_list),
                        tensor,
                        group=gpc.get_group(ParallelMode.TENSOR),
                        async_op=False)
    else:
        tensor_list = [
            torch.empty_like(tensor) for _ in range(gpc.get_world_size(ParallelMode.TENSOR))
        ]
        dist.all_gather(tensor_list,
                        tensor,
                        group=gpc.get_group(ParallelMode.TENSOR),
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
    if gpc.get_world_size(ParallelMode.TENSOR) == 1:
        return tensor

    split_size = divide(tensor.shape[in_dim], gpc.get_world_size(ParallelMode.TENSOR))
    input_tensor_list = torch.split(tensor, split_size, dim=in_dim)

    input_tensor_list = [tensor_.contiguous() for tensor_ in input_tensor_list]
    if out_dim == 1:
        output_shape = list(input_tensor_list[0].shape)
        output_shape[1] *= gpc.get_world_size(ParallelMode.TENSOR)
        output = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
        output_tensor_list = output.chunk(gpc.get_world_size(ParallelMode.TENSOR), dim=1)
        dist.all_to_all(list(output_tensor_list),
                        input_tensor_list,
                        group=gpc.get_group(ParallelMode.TENSOR),
                        async_op=False)
    else:
        output_tensor_list = [torch.ones_like(tensor_) for tensor_ in input_tensor_list]

        dist.all_to_all(output_tensor_list,
                        input_tensor_list,
                        group=gpc.get_group(ParallelMode.TENSOR),
                        async_op=False)

        output = torch.cat(output_tensor_list, dim=out_dim)

    return output


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
