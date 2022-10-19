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
from fastfold.utils.test_utils import get_param_path, get_data_path


config = model_config('model_1')
config.globals.chunk_size = None
config.globals.inplace = False
model_ = AlphaFold(config)
import_jax_weights_(model_, get_param_path())
model_ = model_.eval()
fastmodel_ = copy.deepcopy(model_)
fastmodel_ = inject_fastnn(fastmodel_)
fastmodel_ = fastmodel_.eval()


@pytest.mark.parametrize('world_size', [1, 2])
@pytest.mark.parametrize('chunk_size', [None, 3])
@pytest.mark.parametrize('inplace', [False, True])
def test_state_dict(world_size, chunk_size, inplace):
    run_func = partial(run_dist, world_size=world_size, chunk_size=chunk_size, inplace=inplace)
    mp.spawn(run_func, nprocs=world_size)


def run_dist(rank, world_size, chunk_size, inplace):
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    # init distributed for Dynamic Axial Parallelism
    fastfold.distributed.init_dap()

    global model_, fastmodel_
    model = model_.cuda()
    fastmodel = fastmodel_.cuda()

    set_chunk_size(model.globals.chunk_size)
    batch = pickle.load(open(get_data_path(), 'rb'))
    batch = {k: torch.as_tensor(v).cuda() for k, v in batch.items()}
    fastbatch = copy.deepcopy(batch)

    with torch.no_grad():
        out = model(batch)
        config.globals.inplace = inplace
        fastout = fastmodel(fastbatch)

    pos_dif = torch.max(torch.abs(fastout["final_atom_positions"] - out["final_atom_positions"]))
    assert pos_dif < 1e-3, f"Test failed at chunk size: {chunk_size}, inplace: {inplace}. The position dif is {pos_dif}"
