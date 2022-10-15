import torch
import pytest
import os
import torch.multiprocessing as mp
from functools import partial
import fastfold
from torch import nn
from fastfold.config import model_config
from fastfold.model.nn.evoformer import EvoformerBlock as TargetEvoformer
from fastfold.model.fastnn.evoformer import Evoformer as FastEvoformer
from fastfold.utils.inject_fastnn import copy_evoformer_para
from fastfold.model.fastnn.ops import set_chunk_size


@pytest.mark.parametrize('world_size', [1, 2])
@pytest.mark.parametrize('chunk_size', [None, 3])
@pytest.mark.parametrize('inplace', [False, True])
def test_state_dict(world_size, chunk_size, inplace):
    run_func = partial(_test_evoformer, world_size=world_size, chunk_size=chunk_size, inplace=inplace)
    mp.spawn(run_func, nprocs=world_size)


def _test_evoformer(rank, world_size, chunk_size, inplace):
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    # init distributed for Dynamic Axial Parallelism
    fastfold.distributed.init_dap()
    
    config = model_config('model_1').model["evoformer_stack"]
    with torch.no_grad():
        target_module = TargetEvoformer(
            c_m=config["c_m"],
            c_z=config["c_z"],
            c_hidden_msa_att=config["c_hidden_msa_att"],
            c_hidden_opm=config["c_hidden_opm"],
            c_hidden_mul=config["c_hidden_mul"],
            c_hidden_pair_att=config["c_hidden_pair_att"],
            no_heads_msa=config["no_heads_msa"],
            no_heads_pair=config["no_heads_pair"],
            transition_n=config["transition_n"],
            msa_dropout=config["msa_dropout"],
            pair_dropout=config["pair_dropout"],
            inf=config["inf"],
            eps=config["eps"],
            is_multimer=False,
        )
        fast_module_1 = FastEvoformer(
            c_m=config["c_m"],
            c_z=config["c_z"],
            first_block=True,
            last_block=False,
            is_multimer=False,
        )
        fast_module_2 = FastEvoformer(
            c_m=config["c_m"],
            c_z=config["c_z"],
            first_block=False,
            last_block=True,
            is_multimer=False,
        )
        copy_evoformer_para(fast_module_1, target_module)
        copy_evoformer_para(fast_module_2, target_module)

    target_module = target_module.eval().cuda()
    fast_module_1 = fast_module_1.eval().cuda()
    fast_module_2 = fast_module_2.eval().cuda()

    msa_len = 88
    seq_len = 378
    m = torch.randn((msa_len, seq_len, config["c_m"])).cuda()
    m_mask = torch.ones((msa_len, seq_len)).cuda()
    m_mask[:, :-5] = 0
    z = torch.randn((seq_len, seq_len, config["c_z"])).cuda()
    z_mask = torch.ones((seq_len, seq_len)).cuda()
    z_mask[:, :-5] = 0

    m_out, z_out = target_module(m, z, m_mask, z_mask)
    m_out, z_out = target_module(m_out, z_out, m_mask, z_mask)
    
    if chunk_size:
        set_chunk_size(chunk_size)
    with torch.no_grad():
        if inplace:
            m_fast, z_fast = fast_module_1(m, z, m_mask, z_mask)
            m_fast, z_fast = fast_module_2(m_fast, z_fast, m_mask, z_mask)
        else:
            m_fast, z_fast = fast_module_1.inplace([m], [z], m_mask, z_mask)
            m_fast, z_fast = fast_module_2.inplace(m_fast, z_fast, m_mask, z_mask)
            m_fast = m_fast[0]
            z_fast = z_fast[0]
    assert torch.allclose(m_out, m_fast, atol=1e-8)
    assert torch.allclose(z_out, z_fast, atol=1e-8)
