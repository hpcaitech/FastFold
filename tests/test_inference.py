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
def get_openfold_module_and_data():
    config = model_config('model_1')
    config.globals.inplace = False
    model = AlphaFold(config)
    import_jax_weights_(model, get_param_path())
    model.eval().cuda()
    batch = pickle.load(open(get_data_path(), 'rb'))
    batch = {k: torch.as_tensor(v).cuda() for k, v in batch.items()}
    with torch.no_grad():
        out = model(batch)
        
    fastmodel = copy.deepcopy(model)
    fastmodel = inject_fastnn(fastmodel)
    fastmodel.eval().cuda()
    return model, out, fastmodel


@pytest.mark.parametrize('world_size', [1, 2])
@pytest.mark.parametrize('chunk_size', [None, 32])
@pytest.mark.parametrize('inplace', [False, True])
def test_state_dict(world_size, chunk_size, inplace, get_openfold_module_and_data):
    run_func = partial(run_dist, world_size=world_size, chunk_size=chunk_size, inplace=inplace, model=get_openfold_module_and_data)
    mp.spawn(run_func, nprocs=world_size)


def run_dist(rank, world_size, chunk_size, inplace, model):
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    # init distributed for Dynamic Axial Parallelism
    fastfold.distributed.init_dap()
    inference(chunk_size, inplace, model)


def inference(chunk_size, inplace, get_openfold_module_and_data):

    model, out, fastmodel = get_openfold_module_and_data

    model.globals.chunk_size = chunk_size
    model.globals.inplace = inplace

    fastmodel = copy.deepcopy(fastmodel).cuda()

    fastmodel.structure_module.default_frames = fastmodel.structure_module.default_frames.cuda()
    fastmodel.structure_module.group_idx = fastmodel.structure_module.group_idx.cuda()
    fastmodel.structure_module.atom_mask = fastmodel.structure_module.atom_mask.cuda()
    fastmodel.structure_module.lit_positions = fastmodel.structure_module.lit_positions.cuda()

    set_chunk_size(model.globals.chunk_size)
    batch = pickle.load(open(get_data_path(), 'rb'))
    batch = {k: torch.as_tensor(v).cuda() for k, v in batch.items()}

    with torch.no_grad():
        fastout = fastmodel(batch)

    pos_dif = torch.max(torch.abs(fastout["final_atom_positions"] - out["final_atom_positions"].cuda()))
    assert pos_dif < 5e-4, f"Test failed at chunk size: {chunk_size}, inplace: {inplace}. The position dif is {pos_dif}"
