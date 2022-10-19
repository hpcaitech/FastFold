import os
import copy
import pytest
import torch
import pickle
import torch.multiprocessing as mp
from functools import partial

import fastfold
from fastfold.model.hub import AlphaFold
from fastfold.config import model_config
from fastfold.model.fastnn import set_chunk_size
from fastfold.utils.inject_fastnn import inject_fastnn
from fastfold.utils.import_weights import import_jax_weights_
from fastfold.utils.test_utils import get_data_path, get_param_path


@pytest.fixture(scope="module")
def get_module_and_data():
    config = model_config('model_1')
    config.globals.inplace = False
    model = AlphaFold(config)
    import_jax_weights_(model, get_param_path())
    model.eval()
    model.cuda()
    batch_cpu = pickle.load(open(get_data_path(), 'rb'))
    batch = {k: torch.as_tensor(copy.deepcopy(v)).cuda() for k, v in batch_cpu.items()}
    with torch.no_grad():
        out = model(batch)

    fast_model = copy.deepcopy(model)
    fast_model = inject_fastnn(fast_model)
    fast_model.eval()
    return fast_model, out, batch_cpu


@pytest.mark.parametrize('world_size', [1, 2])
@pytest.mark.parametrize('chunk_size', [None, 1])
@pytest.mark.parametrize('inplace', [False, True])
def test_state_dict(world_size, chunk_size, inplace, get_module_and_data):
    run_func = partial(_test_inference, world_size=world_size, chunk_size=chunk_size, inplace=inplace, get_module_and_data=get_module_and_data)
    mp.spawn(run_func, nprocs=world_size)


def _test_inference(rank, world_size, chunk_size, inplace, get_module_and_data):
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    # init distributed for Dynamic Axial Parallelism
    fastfold.distributed.init_dap()

    fast_model_, openfold_out, batch_ = get_module_and_data

    fast_model = copy.deepcopy(fast_model_)
    fast_model.cuda()

    fast_model.globals.chunk_size = chunk_size
    fast_model.globals.inplace = inplace

    fast_model.structure_module.default_frames = fast_model.structure_module.default_frames.cuda()
    fast_model.structure_module.group_idx = fast_model.structure_module.group_idx.cuda()
    fast_model.structure_module.atom_mask = fast_model.structure_module.atom_mask.cuda()
    fast_model.structure_module.lit_positions = fast_model.structure_module.lit_positions.cuda()

    set_chunk_size(fast_model.globals.chunk_size)
    batch = copy.deepcopy(batch_)
    batch = {k: torch.as_tensor(v).cuda() for k, v in batch.items()}

    with torch.no_grad():
        fast_out = fast_model(batch)

    pos_dif = torch.max(torch.abs(fast_out["final_atom_positions"] - openfold_out["final_atom_positions"].cuda()))
    assert pos_dif < 5e-4, f"Test failed at chunk size: {chunk_size}, inplace: {inplace}. The position dif is {pos_dif}"
