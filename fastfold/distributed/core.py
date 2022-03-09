import torch.distributed as dist

# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None
# Intra-layer model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None

# These values enable us to change the mpu sizes on the fly.
_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
_TENSOR_MODEL_PARALLEL_RANK = None


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(numerator, denominator)


def init_dap(tensor_model_parallel_size_=1):

    assert dist.is_initialized()

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # check dist config
    ensure_divisibility(world_size, tensor_model_parallel_size_)
    data_parallel_size_ = world_size // tensor_model_parallel_size_

    # Build the data-parallel groups.
    global _DATA_PARALLEL_GROUP
    assert _DATA_PARALLEL_GROUP is None, \
        'data parallel group is already initialized'
    for i in range(tensor_model_parallel_size_):
        ranks = range(i, world_size, tensor_model_parallel_size_)
        group = dist.new_group(ranks)
        if rank in ranks:
            _DATA_PARALLEL_GROUP = group

    global _TENSOR_MODEL_PARALLEL_GROUP
    assert _TENSOR_MODEL_PARALLEL_GROUP is None, \
        'tensor model parallel group is already initialized'
    # Build the model-parallel groups.
    for i in range(data_parallel_size_):
        ranks = range(i * tensor_model_parallel_size_, (i + 1) * tensor_model_parallel_size_)
        group = dist.new_group(ranks)
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = group

    if dist.get_rank() == 0:
        print('> initialize tensor model parallel with size {}'.format(tensor_model_parallel_size_))
        print('> initialize data parallel with size {}'.format(data_parallel_size_))


def dap_is_initialized():
    """Check if model and data parallel groups are initialized."""
    if _TENSOR_MODEL_PARALLEL_GROUP is None or \
        _DATA_PARALLEL_GROUP is None:
        return False
    return True


def get_tensor_model_parallel_group():
    """Get the tensor model parallel group the caller rank belongs to."""
    assert _TENSOR_MODEL_PARALLEL_GROUP is not None, \
        'intra_layer_model parallel group is not initialized'
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP is not None, \
        'data parallel group is not initialized'
    return _DATA_PARALLEL_GROUP


def get_tensor_model_parallel_world_size():
    if not dap_is_initialized():
        return 1
    """Return world size for the tensor model parallel group."""
    global _TENSOR_MODEL_PARALLEL_WORLD_SIZE
    if _TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return dist.get_world_size(group=get_tensor_model_parallel_group())


def get_tensor_model_parallel_rank():
    if not dap_is_initialized():
        return 0
    """Return my rank for the tensor model parallel group."""
    global _TENSOR_MODEL_PARALLEL_RANK
    if _TENSOR_MODEL_PARALLEL_RANK is not None:
        return _TENSOR_MODEL_PARALLEL_RANK
    return dist.get_rank(group=get_tensor_model_parallel_group())


def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    return dist.get_world_size(group=get_data_parallel_group())


def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    return dist.get_rank(group=get_data_parallel_group())


def get_tensor_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    global_rank = dist.get_rank()
    local_world_size = get_tensor_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size
