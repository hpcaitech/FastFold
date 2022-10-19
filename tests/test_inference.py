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


@pytest.mark.parametrize('world_size', [1, 2])
@pytest.mark.parametrize('chunk_size', [None, 3])
@pytest.mark.parametrize('inplace', [False, True])
def test_state_dict(world_size, chunk_size, inplace):
    config = model_config('model_1')
    config.globals.chunk_size = chunk_size
    config.globals.inplace = False
    model = AlphaFold(config)
    import_jax_weights_(model, get_param_path())
    model.eval().cuda()
    
    fastmodel = copy.deepcopy(model)
    fastmodel = inject_fastnn(fastmodel)
    fastmodel.eval().cuda()
    
    run_func = partial(run_dist, world_size=world_size, chunk_size=chunk_size, inplace=inplace, model=model, fastmodel=fastmodel, config=config)
    mp.spawn(run_func, nprocs=world_size)


def run_dist(rank, world_size, chunk_size, inplace, model, fastmodel, config):
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    # init distributed for Dynamic Axial Parallelism
    fastfold.distributed.init_dap()

    set_chunk_size(model.globals.chunk_size)
    batch = pickle.load(open(get_data_path(), 'rb'))
    batch = {k: torch.as_tensor(v).cuda() for k, v in batch.items()}
    fastbatch = copy.deepcopy(batch)

    with torch.no_grad():
        out = model(batch)
        config.globals.inplace = inplace
        fastout = fastmodel(fastbatch)

    pos_dif = torch.max(torch.abs(fastout["final_atom_positions"] - out["final_atom_positions"]))
    assert pos_dif < 5e-4, f"Test failed at chunk size: {chunk_size}, inplace: {inplace}. The position dif is {pos_dif}"
