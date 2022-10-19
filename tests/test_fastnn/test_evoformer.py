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


@pytest.mark.parametrize('world_size', [1, 2])
@pytest.mark.parametrize('chunk_size', [None, 3])
@pytest.mark.parametrize('inplace', [False, True])
def test_state_dict(world_size, chunk_size, inplace):
    config = model_config('model_1')
    config.globals.chunk_size = chunk_size
    config.globals.inplace = False
    target_module = AlphaFold(config)
    import_jax_weights_(target_module, get_param_path())

    fast_module = copy.deepcopy(target_module)
    fast_module = inject_fastnn(fast_module)

    target_module = target_module.evoformer
    fast_module = fast_module.evoformer

    target_module_1 = target_module.blocks[0].eval().cuda()
    target_module_2 = target_module.blocks[-1].eval().cuda()
    fast_module_1 = fast_module.blocks[0].eval().cuda()
    fast_module_2 = fast_module.blocks[-1].eval().cuda()
    
    run_func = partial(_test_evoformer, world_size=world_size, chunk_size=chunk_size, 
                       inplace=inplace, fast_module_1=fast_module_1, fast_module_2=fast_module_2, 
                       target_module_1=target_module_1, target_module_2=target_module_2, config=config)
    mp.spawn(run_func, nprocs=world_size)


def _test_evoformer(rank, world_size, chunk_size, inplace, fast_module_1, fast_module_2, 
                    target_module_1, target_module_2, config):
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    # init distributed for Dynamic Axial Parallelism
    fastfold.distributed.init_dap()

    msa_len = 80
    seq_len = 80
    
    target_module_1 = target_module_1.cuda()
    target_module_2 = target_module_2.cuda()
    fast_module_1 = fast_module_1.cuda()
    fast_module_2 = fast_module_2.cuda()

    m = torch.randn((msa_len, seq_len, 256)).cuda()
    m_mask = torch.ones((msa_len, seq_len)).cuda()
    z = torch.randn((seq_len, seq_len, 128)).cuda()
    z_mask = torch.ones((seq_len, seq_len)).cuda()

    set_chunk_size(chunk_size)
    with torch.no_grad():
        m_out, z_out = target_module_1(m, z, m_mask, z_mask)
        m_out, z_out = target_module_2(m_out, z_out, m_mask, z_mask)

        if not inplace:
            m_fast, z_fast = fast_module_1(m, z, m_mask, z_mask)
            m_fast, z_fast = fast_module_2(m_fast, z_fast, m_mask, z_mask)
        else:
            m_fast, z_fast = fast_module_1.inplace([m], [z], m_mask, z_mask)
            m_fast, z_fast = fast_module_2.inplace(m_fast, z_fast, m_mask, z_mask)
            m_fast = m_fast[0]
            z_fast = z_fast[0]

    error = torch.mean(torch.abs(m_out - m_fast))
    assert error < 1e-4, f"Test m failed at chunk size: {chunk_size}, inplace: {inplace}. The position dif is {error}"
    error = torch.mean(torch.abs(z_out - z_fast))
    assert error < 1e-4, f"Test z failed at chunk size: {chunk_size}, inplace: {inplace}. The position dif is {error}"
