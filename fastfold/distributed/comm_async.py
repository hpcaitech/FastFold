from typing import Tuple
from einops import rearrange

import torch
import torch.distributed as dist
from torch import Tensor

from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc

from .comm import _split, divide


def broadcast_sync(src: int, tensor: Tensor, host: bool = False) -> Tensor:
    if gpc.get_world_size(ParallelMode.TENSOR) == 1:
        return 0

    if host:
        dist.broadcast(tensor,
                            src=src,
                            group=gpc.get_group(ParallelMode.TENSOR),
                            async_op=False)
        return 0

    else:
        output = torch.empty(list(tensor.shape), dtype=tensor.dtype, device=tensor.device)
        dist.broadcast(output,
                            src=src,
                            group=gpc.get_group(ParallelMode.TENSOR),
                            async_op=False)
        return output


def broadcast_async(src: int, tensor: Tensor, host: bool = False) -> Tensor:
    if gpc.get_world_size(ParallelMode.TENSOR) == 1:
        return 0

    if host:
        work = dist.broadcast(tensor,
                            src=src,
                            group=gpc.get_group(ParallelMode.TENSOR),
                            async_op=True)
        return work

    else:
        work = dist.broadcast(tensor,
                            src=src,
                            group=gpc.get_group(ParallelMode.TENSOR),
                            async_op=True)
        return work


def broadcast_async_opp(work) -> Tensor:
    work.wait()
    return 0


def get_rank():
    return gpc.get_global_rank()


def get_world_size():
    return gpc.get_world_size(ParallelMode.TENSOR)


def _gather_async(tensor: Tensor, dim: int = -1) -> Tensor:
    if gpc.get_world_size(ParallelMode.TENSOR) == 1:
        return tensor, None

    output_shape = list(tensor.shape)
    output_shape[1] *= gpc.get_world_size(ParallelMode.TENSOR)
    output = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
    tensor_list = output.chunk(gpc.get_world_size(ParallelMode.TENSOR), dim=1)
    work = dist.all_gather(list(tensor_list),
                           tensor,
                           group=gpc.get_group(ParallelMode.TENSOR),
                           async_op=True)

    return output, work


def gather_async(input: Tensor, dim: int = -1) -> Tensor:
    if torch.is_grad_enabled() and input.requires_grad:
        input, work = GatherAsync.apply(input, dim)
    else:
        input, work = _gather_async(input, dim=dim)
    return input, work


def gather_async_opp(output: Tensor, work, dim: int = -1) -> Tensor:
    if work:
        work.wait()
    if dim == 2:
        output = GatherAsyncOpp.apply(output)
    return output


class GatherAsyncOpp(torch.autograd.Function):

    @staticmethod
    def forward(ctx: "GatherAsyncOpp", input: Tensor) -> Tensor:
        mp_size = gpc.get_world_size(ParallelMode.TENSOR)
        output = rearrange(input, 'n (x h) w c -> n h (x w) c', x=mp_size)
        return output

    @staticmethod
    def backward(ctx: "GatherAsyncOpp", grad_output: Tensor) -> Tuple[Tensor]:
        mp_size = gpc.get_world_size(ParallelMode.TENSOR)
        n, h, w, c = grad_output.shape
        return grad_output.resize_(n, h * mp_size, int(w / mp_size), c)


class GatherAsync(torch.autograd.Function):

    @staticmethod
    def forward(ctx: "GatherAsync", input: Tensor, dim: int = -1) -> Tensor:
        ctx.dim = dim
        return _gather_async(input, dim=dim)

    @staticmethod
    def backward(ctx: "GatherAsync", grad_output: Tensor, grad_work=None) -> Tuple[Tensor]:
        if ctx.dim == 2:
            mp_size = gpc.get_world_size(ParallelMode.TENSOR)
            n, h, w, c = grad_output.shape
            grad_output.resize_(n, int(h / mp_size), w * mp_size, c)
        return _split(grad_output, dim=ctx.dim), None


def _all_to_all_async(tensor: Tensor, in_dim: int = -1, out_dim: int = -1) -> Tensor:
    if gpc.get_world_size(ParallelMode.TENSOR) == 1:
        return tensor, None

    split_size = divide(tensor.shape[in_dim], gpc.get_world_size(ParallelMode.TENSOR))
    input_tensor_list = torch.split(tensor, split_size, dim=in_dim)

    input_tensor_list = [tensor_.contiguous() for tensor_ in input_tensor_list]

    output_shape = list(input_tensor_list[0].shape)
    output_shape[1] *= gpc.get_world_size(ParallelMode.TENSOR)
    output = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
    output_tensor_list = output.chunk(gpc.get_world_size(ParallelMode.TENSOR), dim=1)
    work = dist.all_to_all(list(output_tensor_list),
                           input_tensor_list,
                           group=gpc.get_group(ParallelMode.TENSOR),
                           async_op=True)

    return output, work


WORLD_WORK_ALL2ALL = None


class All_to_All_Async(torch.autograd.Function):

    @staticmethod
    def forward(ctx: "All_to_All_Async",
                input_: Tensor,
                in_dim: int = -1,
                out_dim: int = -1) -> Tensor:
        ctx.in_dim = in_dim
        ctx.out_dim = out_dim
        return _all_to_all_async(input_, in_dim=in_dim, out_dim=out_dim)

    @staticmethod
    def backward(ctx: "All_to_All_Async", grad_output: Tensor, grad_work=None) -> Tuple[Tensor]:
        global WORLD_WORK_ALL2ALL
        if WORLD_WORK_ALL2ALL:
            WORLD_WORK_ALL2ALL.wait()
        WORLD_WORK_ALL2ALL = None
        if ctx.in_dim == 2:
            mp_size = gpc.get_world_size(ParallelMode.TENSOR)
            grad_output = rearrange(grad_output, 'n (x h) w c -> n h (x w) c', x=mp_size)
        return grad_output, None, None


class All_to_All_Async_Opp(torch.autograd.Function):

    @staticmethod
    def forward(ctx: "All_to_All_Async_Opp",
                output: Tensor,
                work,
                in_dim: int = -1,
                out_dim: int = -1) -> Tensor:
        ctx.in_dim = in_dim
        ctx.out_dim = out_dim
        if work:
            work.wait()
        if out_dim == 2:
            mp_size = gpc.get_world_size(ParallelMode.TENSOR)
            output = rearrange(output, 'n (x h) w c -> n h (x w) c', x=mp_size)
        return output

    @staticmethod
    def backward(ctx: "All_to_All_Async_Opp", grad_output: Tensor) -> Tuple[Tensor]:
        global WORLD_WORK_ALL2ALL
        d_tensor, WORLD_WORK_ALL2ALL = _all_to_all_async(grad_output,
                                                         in_dim=ctx.out_dim,
                                                         out_dim=ctx.in_dim)
        return d_tensor, None, None, None