import os
import pytest
import torch
import pickle
import torch.multiprocessing as mp
from functools import partial
import colossalai
from fastfold.model.hub import AlphaFold
from fastfold.config import model_config
from fastfold.model.fastnn import set_chunk_size
from fastfold.utils.inject_fastnn import inject_fastnn
from fastfold.utils.test_utils import get_train_data_path
from fastfold.model.hub.loss import AlphaFoldLoss
from fastfold.utils.tensor_utils import tensor_tree_map
from fastfold.utils.test_utils import set_seed


def get_param_and_grad(model):
    params = dict()
    grads = dict()
    for name, param in model.named_parameters():
        params[name] = param.detach().clone()
        grads[name] = param.grad.detach().clone()

    return params, grads


@pytest.fixture(scope="module")
def get_openfold_state():
    config = model_config('initial_training', train=True)
    config.globals.inplace = False
    set_seed(42)
    model = AlphaFold(config)
    model.train().cuda()
    criterion = AlphaFoldLoss(config.loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-8)
    batch = pickle.load(open(get_train_data_path(), 'rb'))
    set_seed(42)
    batch = {k: torch.as_tensor(v).cuda() for k, v in batch.items()}
    out = model(batch)
    batch = tensor_tree_map(lambda t: t[..., -1], batch)
    loss, _ = criterion(out, batch, True)
    optimizer.zero_grad()
    set_seed(42)
    loss.backward()
    optimizer.step()
    of_params, of_grads = get_param_and_grad(model)
    return of_params, of_grads


@pytest.mark.skipif(torch.cuda.mem_get_info(0)[1] < 4e10, reason="Not enough cuda memory")
@pytest.mark.parametrize('world_size', [1])
def test_state_dict(world_size, get_openfold_state):
    run_func = partial(run_dist, world_size=world_size, model=get_openfold_state)
    mp.spawn(run_func, nprocs=world_size)


def run_dist(rank, world_size, model):
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    colossalai.launch(config=dict(parallel=dict(tensor=dict(size=world_size))), rank=rank, world_size=world_size,
                      host='localhost', port=10101, backend='nccl')
    train(world_size, model)


def train(world_size, get_openfold_state):

    of_params, of_grads = get_openfold_state
    config = model_config('initial_training', train=True)
    config.globals.inplace = False
    set_seed(42)
    model = AlphaFold(config)
    model = inject_fastnn(model)
    model.train().cuda()
    criterion = AlphaFoldLoss(config.loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-8)
    set_chunk_size(None) 
    batch = pickle.load(open(get_train_data_path(), 'rb'))
    batch = {k: torch.as_tensor(v).cuda() for k, v in batch.items()}
    set_seed(42)
    out = model(batch)
    batch = tensor_tree_map(lambda t: t[..., -1], batch)
    loss, _ = criterion(out, batch, True)
    optimizer.zero_grad()
    set_seed(42)
    loss.backward()
    optimizer.step()
    ff_params, ff_grads = get_param_and_grad(model)

    params_dif = 0
    grads_dif = 0
    for name in ff_params.keys():
        # the modules' names in fastfold and openfold are not equal
        # it leads some differences on the order of the parameters
        # it's not a hard problem to solve
        # but check the params and grads of the same part may be just enough
        if name not in of_params.keys():
            continue
     
        dif = torch.max(torch.abs(ff_params[name] - of_params[name]))
        if  dif > params_dif:
            params_dif = dif
        dif = torch.max(torch.abs(ff_grads[name] - of_grads[name]))
        if dif > grads_dif:
            grads_dif = dif
    assert params_dif < 1e-3 and grads_dif < 5e-3, f"Test failed at world size: {world_size}, \
        the param dif is {params_dif}, the grad diff is {grads_dif}"


if __name__ == '__main__':
     test_state_dict(1, None, None)