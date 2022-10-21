import torch
import pytest
import os
import copy
import torch.multiprocessing as mp
from functools import partial
import fastfold
from fastfold.config import model_config
from fastfold.model.fastnn.ops import set_chunk_size
from fastfold.model.hub import AlphaFold
from fastfold.utils.inject_fastnn import inject_fastnn
from fastfold.utils.import_weights import import_jax_weights_
from fastfold.utils.test_utils import get_param_path


@pytest.fixture(scope="module")
def get_openfold_module_and_data():
    with torch.no_grad():
        config = model_config('model_1')
        config.globals.inplace = False
        
        target_module = AlphaFold(config)
        import_jax_weights_(target_module, get_param_path())
        fast_module = copy.deepcopy(target_module)
        fast_module = inject_fastnn(fast_module)
        fast_module = fast_module.extra_msa_stack
        fast_module = fast_module.cuda().eval()
        
        extra_msa_len = 300
        seq_len = 64
        m = torch.randn((extra_msa_len, seq_len, 64)).cuda()
        m_mask = torch.ones((extra_msa_len, seq_len)).cuda().to(dtype=m.dtype)
        m_mask[64:, :] = 0.
        z = torch.randn((seq_len, seq_len, 128)).cuda()
        z_mask = torch.ones((seq_len, seq_len)).cuda().to(dtype=z.dtype)
        data = [m, z, m_mask, z_mask]
        inputs = [copy.deepcopy(i).cuda() for i in data]
        
        target_module = target_module.extra_msa_stack
        target_module = target_module.eval().cuda()
        z_out = target_module(
            inputs[0], inputs[1], msa_mask=inputs[2], pair_mask=inputs[3], chunk_size=None, _mask_trans=config.model._mask_trans)

    return z_out, config, fast_module, data


@pytest.mark.parametrize('world_size', [1, 2])
@pytest.mark.parametrize('chunk_size', [None, 32])
@pytest.mark.parametrize('inplace', [False, True])
def test_state_dict(world_size, chunk_size, inplace, get_openfold_module_and_data):
    run_func = partial(_test_extramsa_stack, world_size=world_size, chunk_size=chunk_size, inplace=inplace, 
                       get_openfold_module_and_data=get_openfold_module_and_data)
    mp.spawn(run_func, nprocs=world_size)


def _test_extramsa_stack(rank, world_size, chunk_size, inplace, get_openfold_module_and_data):
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    # init distributed for Dynamic Axial Parallelism
    fastfold.distributed.init_dap()

    z_out, config, fast_module, data = get_openfold_module_and_data
    inputs = [copy.deepcopy(i).cuda() for i in data]
    fast_module = copy.deepcopy(fast_module).eval().cuda()

    with torch.no_grad():
        set_chunk_size(chunk_size)
        if not inplace:
            z_fast = fast_module(
                inputs[0], inputs[1], msa_mask=inputs[2], pair_mask=inputs[3], chunk_size=chunk_size, _mask_trans=config.model._mask_trans)
        else:
            z_fast = fast_module.inplace(
                [inputs[0]], [inputs[1]], msa_mask=inputs[2], pair_mask=inputs[3], chunk_size=chunk_size, _mask_trans=config.model._mask_trans)
            z_fast = z_fast[0]

    error = torch.mean(torch.abs(z_out.cuda() - z_fast))
    assert error < 1e-3, f"Test z failed at chunk size: {chunk_size}, inplace: {inplace}. The position dif is {error}"
