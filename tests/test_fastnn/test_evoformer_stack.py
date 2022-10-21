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
def get_module_and_output():
    with torch.no_grad():
        config = model_config('model_1')
        config.globals.inplace = False
        model = AlphaFold(config)
        import_jax_weights_(model, get_param_path())

        fast_model = copy.deepcopy(model)
        fast_model = inject_fastnn(fast_model)
        fast_model = fast_model.evoformer
        fast_model.eval().cuda()

        model = model.evoformer
        model.eval().cuda()
        
        msa_len = 50
        seq_len = 52
        m = torch.randn((msa_len, seq_len, 256))
        m_mask = torch.ones((msa_len, seq_len)).to(dtype=m.dtype)
        z = torch.randn((seq_len, seq_len, 128))
        z_mask = torch.ones((seq_len, seq_len)).to(dtype=z.dtype)
        data = [m, z, m_mask, z_mask]
        inputs = [copy.deepcopy(i).cuda() for i in data]
        out = model(
            *inputs, chunk_size=None, _mask_trans=config.model._mask_trans)
    return fast_model, config, out, data


@pytest.mark.parametrize('world_size', [1, 2])
@pytest.mark.parametrize('chunk_size', [None, 1])
@pytest.mark.parametrize('inplace', [False, True])
def test_state_dict(world_size, chunk_size, inplace, get_module_and_output):
    run_func = partial(_test_evoformer_stack, world_size=world_size, chunk_size=chunk_size, 
                       inplace=inplace, get_module_and_output=get_module_and_output)
    mp.spawn(run_func, nprocs=world_size)


def _test_evoformer_stack(rank, world_size, chunk_size, inplace, get_module_and_output):
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    # init distributed for Dynamic Axial Parallelism
    fastfold.distributed.init_dap()

    fast_module, config, out, data = get_module_and_output
    inputs = [copy.deepcopy(i).cuda() for i in data]
    fast_module = copy.deepcopy(fast_module).eval().cuda()

    with torch.no_grad():
        set_chunk_size(chunk_size)
        if not inplace:
            m_fast, z_fast, s_fast = fast_module(
                *inputs, chunk_size=chunk_size, _mask_trans=config.model._mask_trans)
        else:
            m_fast, z_fast, s_fast = fast_module.inplace(
                [inputs[0]], [inputs[1]], inputs[2], inputs[3], chunk_size=chunk_size, _mask_trans=config.model._mask_trans)
            m_fast = m_fast[0]
            z_fast = z_fast[0]

    error = torch.mean(torch.abs(out[0].cuda() - m_fast))
    assert error < 2e-3, f"Test m failed at chunk size: {chunk_size}, inplace: {inplace}. The position dif is {error}"
    error = torch.mean(torch.abs(out[1].cuda() - z_fast))
    assert error < 2e-3, f"Test z failed at chunk size: {chunk_size}, inplace: {inplace}. The position dif is {error}"
    error = torch.mean(torch.abs(out[2].cuda() - s_fast))
    assert error < 2e-3, f"Test s failed at chunk size: {chunk_size}, inplace: {inplace}. The position dif is {error}"
