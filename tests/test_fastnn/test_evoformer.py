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
        target_module = AlphaFold(config)
        import_jax_weights_(target_module, get_param_path())

        fast_module = copy.deepcopy(target_module)
        fast_module = inject_fastnn(fast_module)
        fast_module = fast_module.evoformer
        fast_module_1 = fast_module.blocks[0].eval().cuda()
        fast_module_2 = fast_module.blocks[-1].eval().cuda()

        target_module = target_module.evoformer
        target_module_1 = target_module.blocks[0].eval().cuda()
        target_module_2 = target_module.blocks[-1].eval().cuda()
        
        msa_len = 80
        seq_len = 80
        m = torch.randn((msa_len, seq_len, 256))
        m_mask = torch.ones((msa_len, seq_len))
        z = torch.randn((seq_len, seq_len, 128))
        z_mask = torch.ones((seq_len, seq_len))
        data = [m, z, m_mask, z_mask]
        inputs = [copy.deepcopy(i).cuda() for i in data]
        
        m_out, z_out = target_module_1(*inputs)
        m_out, z_out = target_module_2(m_out, z_out, inputs[2], inputs[3])

    return fast_module_1, fast_module_2, m_out, z_out, data


@pytest.mark.parametrize('world_size', [1, 2])
@pytest.mark.parametrize('chunk_size', [None, 32])
@pytest.mark.parametrize('inplace', [False, True])
def test_state_dict(world_size, chunk_size, inplace, get_module_and_output):
    run_func = partial(_test_evoformer, world_size=world_size, chunk_size=chunk_size, inplace=inplace, get_module_and_output=get_module_and_output)
    mp.spawn(run_func, nprocs=world_size)


def _test_evoformer(rank, world_size, chunk_size, inplace, get_module_and_output):
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    # init distributed for Dynamic Axial Parallelism
    fastfold.distributed.init_dap()

    fast_module_1, fast_module_2, m_out, z_out, data = get_module_and_output
    
    fast_module_1 = copy.deepcopy(fast_module_1).eval().cuda()
    fast_module_2 = copy.deepcopy(fast_module_2).eval().cuda()
    inputs = [copy.deepcopy(i).cuda() for i in data]
    
    set_chunk_size(chunk_size)
    with torch.no_grad():
        if not inplace:
            m_fast, z_fast = fast_module_1(*inputs)
            m_fast, z_fast = fast_module_2(m_fast, z_fast, inputs[2], inputs[3])
        else:
            m_fast, z_fast = fast_module_1.inplace([inputs[0]], [inputs[1]], inputs[2], inputs[3])
            m_fast, z_fast = fast_module_2.inplace(m_fast, z_fast, inputs[2], inputs[3])
            m_fast = m_fast[0]
            z_fast = z_fast[0]

    error = torch.mean(torch.abs(m_out.cuda() - m_fast))
    assert error < 5e-4, f"Test m failed at chunk size: {chunk_size}, inplace: {inplace}. The position dif is {error}"
    error = torch.mean(torch.abs(z_out.cuda() - z_fast))
    assert error < 5e-4, f"Test z failed at chunk size: {chunk_size}, inplace: {inplace}. The position dif is {error}"
