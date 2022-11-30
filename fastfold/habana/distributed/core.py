import os

from mpi4py import MPI

import torch
import torch.distributed as dist


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(numerator, denominator)


def set_missing_distributed_environ(key, value):
    if key not in os.environ:
        os.environ[str(key)] = str(value)


def init_dap(tensor_model_parallel_size_=None):
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank = comm.Get_rank()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12340'

    import habana_frameworks.torch.distributed.hccl
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)
