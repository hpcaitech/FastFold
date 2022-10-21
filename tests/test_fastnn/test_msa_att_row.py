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
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from fastfold.utils.test_utils import get_param_path
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
        fast_module = fast_module.evoformer.blocks[0].msa.MSARowAttentionWithPairBias.eval().cuda()
        target_module1 = target_module.evoformer.blocks[0].msa_att_row.eval().cuda()
        target_module2 = target_module.evoformer.blocks[0].msa_dropout_layer.eval().cuda()
        
        msa_len = 300
        seq_len = 300
        m = torch.randn((msa_len, seq_len, 256)).cuda()
        m_mask = torch.ones((msa_len, seq_len)).cuda().to(dtype=m.dtype)
        z = torch.randn((seq_len, seq_len, 128)).cuda()
        z_mask = torch.ones((seq_len, seq_len)).cuda().to(dtype=z.dtype)
        m_out = m + target_module2(target_module1(m, z=z, mask=m_mask, chunk_size=None))
        
        
    return m_out, m, z, m_mask, z_mask, fast_module


@pytest.mark.parametrize('world_size', [1, 2])
@pytest.mark.parametrize('chunk_size', [None, 32])
def test_state_dict(world_size, chunk_size, get_openfold_module_and_data):
    run_func = partial(_test_msa_att_row, world_size=world_size, chunk_size=chunk_size, get_openfold_module_and_data=get_openfold_module_and_data)
    mp.spawn(run_func, nprocs=world_size)


def _test_msa_att_row(rank, world_size, chunk_size, get_openfold_module_and_data):
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    # init distributed for Dynamic Axial Parallelism
    fastfold.distributed.init_dap()

    m_out, m, z, m_mask, z_mask, fast_module = get_openfold_module_and_data
    fast_module = copy.deepcopy(fast_module).cuda()

    fast_m = copy.deepcopy(m.cuda()).unsqueeze(0)
    fast_z = copy.deepcopy(z.cuda()).unsqueeze(0)
    dap_size = gpc.get_world_size(ParallelMode.TENSOR)
    seq_length = z_mask.cuda().size(-1)
    padding_size = (int(seq_length / dap_size) + 1) * dap_size - seq_length
    fast_m = torch.nn.functional.pad(fast_m, (0, 0, 0, padding_size))
    fast_z = torch.nn.functional.pad(fast_z, (0, 0, 0, padding_size, 0, padding_size))
    fast_m = scatter(fast_m, dim=1)
    fast_z = scatter(fast_z, dim=1)
    fast_m_mask = copy.deepcopy(m_mask.cuda()).unsqueeze(0)
    fast_m_mask = torch.nn.functional.pad(fast_m_mask, (0, padding_size))
    
    with torch.no_grad():
        set_chunk_size(chunk_size)
        fast_m_mask = scatter(fast_m_mask.cuda(), dim=1)
        m_fast = fast_module(fast_m.cuda(), fast_z.cuda(), fast_m_mask)
        m_fast = m_fast.squeeze(0)
        m_fast = gather(m_fast, dim=0)
        m_fast = m_fast[:, :-padding_size, :]

    error = torch.max(torch.abs(m_out.cuda() - m_fast))
    assert error < 5e-5, f"Test m failed at chunk size: {chunk_size}. The position dif is {error}"
