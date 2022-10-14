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
from fastfold.utils import inject_fastnn
from fastfold.utils.import_weights import import_jax_weights_


@pytest.mark.parametrize('world_size', [1, 2])
@pytest.mark.parametrize('chunk_size', [None, 2])
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
    inference(chunk_size, inplace)
    


def inference(chunk_size, inplace):

    config = model_config('model_1')
    config.globals.chunk_size = chunk_size
    config.globals.inplace = False
    model = AlphaFold(config)
    import_jax_weights_(model, '/data/scratch/fastfold/weight.npz')
    model.eval()
    model.cuda()

    fastmodel = copy.deepcopy(model)
    fastmodel = inject_fastnn(fastmodel)
    fastmodel.eval()
    fastmodel.cuda()

    set_chunk_size(model.globals.chunk_size)
    batch = pickle.load(open('/data/scratch/fastfold/mono_batch.pkl', 'rb'))
    batch = {k: torch.as_tensor(v).cuda() for k, v in batch.items()}
    fastbatch = copy.deepcopy(batch)

    with torch.no_grad():
        out = model(batch)
        config.globals.inplace = inplace
        fastout = fastmodel(fastbatch)

    pos_dif = torch.max(torch.abs(fastout["final_atom_positions"] - out["final_atom_positions"]))
    assert pos_dif < 1.1, f"Test failed at chunk size: {chunk_size}, inplace: {inplace}. The position dif is {pos_dif}"
