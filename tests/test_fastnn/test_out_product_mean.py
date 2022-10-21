import torch
import pytest
import copy
import fastfold
import os
import torch.multiprocessing as mp
from functools import partial
from fastfold.model.fastnn.ops import set_chunk_size
from fastfold.utils.test_utils import get_param_path
from fastfold.model.hub import AlphaFold
from fastfold.utils.inject_fastnn import inject_fastnn
from fastfold.utils.import_weights import import_jax_weights_
from fastfold.config import model_config
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from fastfold.utils.test_utils import get_param_path
from fastfold.distributed import scatter, row_to_col
from fastfold.distributed.comm import gather, scatter


@pytest.fixture(scope="module")
def get_openfold_module_and_data():
    with torch.no_grad():
        config = model_config('model_1')
        config.globals.inplace = False
        target_module = AlphaFold(config)
        import_jax_weights_(target_module, get_param_path())

        fast_module = copy.deepcopy(target_module)
        fast_module = inject_fastnn(fast_module)
        fast_module = fast_module.evoformer.blocks[0].communication.eval().cuda()
        target_module = target_module.evoformer.blocks[0].core.outer_product_mean.eval().cuda()

        msa_len = 20
        seq_len = 30
        m = torch.randn((msa_len, seq_len, 256)).cuda()
        m_mask = torch.ones((msa_len, seq_len)).cuda()
        m_mask[:, -5:] = 0
        z = torch.zeros((seq_len, seq_len, 128)).cuda()

        out = target_module(m, m_mask)
    return m, m_mask, z, fast_module, out


@pytest.mark.parametrize('world_size', [1, 2])
@pytest.mark.parametrize('chunk_size', [None, 32])
@pytest.mark.parametrize('inplace', [False, True])
def test_state_dict(world_size, chunk_size, inplace, get_openfold_module_and_data):
    run_func = partial(_test_out_product_mean, world_size=world_size, chunk_size=chunk_size, 
                       inplace=inplace, get_openfold_module_and_data=get_openfold_module_and_data)
    mp.spawn(run_func, nprocs=world_size)


def _test_out_product_mean(rank, world_size, chunk_size, inplace, get_openfold_module_and_data):
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    # init distributed for Dynamic Axial Parallelism
    fastfold.distributed.init_dap()
    
    m, m_mask, z, fast_module, out = get_openfold_module_and_data
    fast_module = copy.deepcopy(fast_module).cuda()
    
    fast_m = copy.deepcopy(m.cuda()).unsqueeze(0)
    fast_z = copy.deepcopy(z.cuda()).unsqueeze(0)
    dap_size = gpc.get_world_size(ParallelMode.TENSOR)
    seq_length = m_mask.cuda().size(-1)
    padding_size = (int(seq_length / dap_size) + 1) * dap_size - seq_length
    fast_m = torch.nn.functional.pad(fast_m, (0, 0, 0, padding_size))
    fast_z = torch.nn.functional.pad(fast_z, (0, 0, 0, padding_size, 0, padding_size))
    fast_m = scatter(fast_m, dim=1)
    fast_z = scatter(fast_z, dim=1)
    fast_m_mask = copy.deepcopy(m_mask.cuda()).unsqueeze(0)
    fast_m_mask = torch.nn.functional.pad(fast_m_mask, (0, padding_size))
    
    with torch.no_grad():
        set_chunk_size(chunk_size)
        fast_m = row_to_col(fast_m)
        if inplace:
            out_fast = fast_module.inplace(fast_m, fast_m_mask, [fast_z])[0]
        else:
            out_fast = fast_module(fast_m, fast_m_mask, fast_z)
        out_fast = out_fast.squeeze(0)
        out_fast = gather(out_fast, dim=0)
        out_fast = out_fast[:-padding_size, :-padding_size, :]
    error = torch.mean(torch.abs(out.cuda() - out_fast))
    assert error < 1e-5, f"Test failed at chunk size: {chunk_size}, inplace: {inplace}. The position dif is {error}"
