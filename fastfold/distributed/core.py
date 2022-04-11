import os

import torch
import colossalai


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(numerator, denominator)

def set_distributed_environ(key, value):
    os.environ[str(key)] = str(value)


def init_dap(tensor_model_parallel_size_=None):
    colossalai.logging.disable_existing_loggers()

    if tensor_model_parallel_size_ == None:
        if 'WORLD_SIZE' in os.environ:
            tensor_model_parallel_size_ = int(os.environ['WORLD_SIZE'])
        else:
            tensor_model_parallel_size_ = 1

    if torch.torch.distributed.is_initialized():
        _logger = colossalai.logging.get_dist_logger()
        _logger.error(
            "use fastfold.distributed.init_dap instead of torch.distributed.init_process_group!")
        exit(-1)

    # set distributed environ for single device launch
    if 'RANK' not in os.environ:
        set_distributed_environ('WORLD_SIZE', 1)
        set_distributed_environ('RANK', 0)
        set_distributed_environ('LOCAL_RANK', 0)
        set_distributed_environ('MASTER_ADDR', "localhost")
        set_distributed_environ('MASTER_PORT', 10045)

    colossalai.launch_from_torch(
        config={"parallel": dict(tensor=dict(size=tensor_model_parallel_size_))})
