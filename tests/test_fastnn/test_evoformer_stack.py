import torch
import pytest
import os
import torch.multiprocessing as mp
from functools import partial
import fastfold
from fastfold.config import model_config
from fastfold.model.nn.evoformer import EvoformerStack as TargetEvoformerStack
from fastfold.model.fastnn.evoformer import EvoformerStack as FastEvoformerStack
from fastfold.utils.inject_fastnn import copy_evoformer_para, copy_linear
from fastfold.model.fastnn.ops import set_chunk_size


@pytest.mark.parametrize('world_size', [1, 2])
@pytest.mark.parametrize('chunk_size', [None, 3])
@pytest.mark.parametrize('inplace', [False, True])
def test_state_dict(world_size, chunk_size, inplace):
    run_func = partial(_test_evoformer_stack, world_size=world_size, chunk_size=chunk_size, inplace=inplace)
    mp.spawn(run_func, nprocs=world_size)


def _test_evoformer_stack(rank, world_size, chunk_size, inplace):
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    # init distributed for Dynamic Axial Parallelism
    fastfold.distributed.init_dap()
    
    config = model_config('model_1').model["evoformer_stack"]
    with torch.no_grad():
        target_module = TargetEvoformerStack(
            is_multimer=False,
            **config,
        )
        fast_module = FastEvoformerStack(
            c_m=target_module.blocks[0].msa_att_row.c_in,
            c_z=target_module.blocks[0].msa_att_row.c_z,
            c_s=target_module.linear.out_features,
            no_blocks=len(target_module.blocks),
            blocks_per_ckpt=target_module.blocks_per_ckpt,
            clear_cache_between_blocks=target_module.clear_cache_between_blocks,
            is_multimer=target_module.blocks[0].is_multimer,
        )
        for target_block, fast_block in zip(target_module.blocks, fast_module.blocks):
            copy_evoformer_para(fast_block, target_block)
            if target_block.training == False:
                fast_block.eval()
        copy_linear(fast_module.linear, target_module.linear)

    target_module = target_module.eval().cuda()
    fast_module = fast_module.eval().cuda()

    msa_len = 60
    seq_len = 120
    m = torch.randn((msa_len, seq_len, config["c_m"])).cuda()
    m_mask = torch.ones((msa_len, seq_len)).cuda()
    m_mask[:, :-5] = 0
    z = torch.randn((seq_len, seq_len, config["c_z"])).cuda()
    z_mask = torch.ones((seq_len, seq_len)).cuda()
    z_mask[:, :-5] = 0

    m_out, z_out, s_out = target_module(m, z, m_mask, z_mask, chunk_size=chunk_size)
    
    if chunk_size:
        set_chunk_size(chunk_size)
    with torch.no_grad():
        if inplace:
            m_fast, z_fast, s_fast = fast_module(m, z, m_mask, z_mask, chunk_size=chunk_size)
        else:
            m_fast, z_fast, s_fast = fast_module.inplace([m], [z], m_mask, z_mask, chunk_size=chunk_size)
            m_fast = m_fast[0]
            z_fast = z_fast[0]
    assert torch.allclose(m_out, m_fast, atol=1e-8)
    assert torch.allclose(z_out, z_fast, atol=1e-8)
    assert torch.allclose(s_out, s_fast, atol=1e-8)
