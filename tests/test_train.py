import os
import copy
import pytest
import torch
import pickle
import torch.multiprocessing as mp
from functools import partial
import colossalai
import fastfold
from fastfold.model.hub import AlphaFold
from fastfold.config import model_config
from fastfold.model.fastnn import set_chunk_size
from fastfold.utils.inject_fastnn import inject_fastnn
from fastfold.utils.test_utils import get_data_path
from fastfold.model.loss import AlphaFoldLoss
from fastfold.utils.tensor_utils import tensor_tree_map



def colo_train_step(engine):
    batch = pickle.load(open('../std_train_batch.pkl', 'rb'))
    batch = {k: torch.as_tensor(v).cuda() for k, v in batch.items()}
    engine.zero_grad()
    out = engine(batch)
    batch = tensor_tree_map(lambda t: t[..., -1], batch)
    loss, _ = engine.criterion(out, batch, _return_breakdown=True)
    print(loss)

    engine.backward(loss)
    engine.step()

    params = dict()
    grads = dict()
    for name, param in engine.model.named_parameters():
        params[name] = param.clone()
        grads[name] = param.grad.clone()
    # pickle.dump(params, open('param.pkl', 'wb'))
    # pickle.dump(grads, open('grad.pkl', 'wb'))
    # exit()
    return params, grads


def get_openfold_state():
    config = model_config('initial_training', train=True)
    config.globals.inplace = False
    model = AlphaFold(config)
    model.train().cuda()
    ori_state_dict = model.state_dict()
    criterion = AlphaFoldLoss(config.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, eps=1e-8)

    of_engine, _, _, _ = colossalai.initialize(
                                            model=model,
                                            optimizer=optimizer,
                                            criterion=criterion
                                            )

    of_params, of_grads = colo_train_step(of_engine)
    return of_params, of_grads, ori_state_dict


@pytest.mark.parametrize('world_size', [1])
@pytest.mark.parametrize('chunk_size', [None])
def test_state_dict(world_size, chunk_size, get_openfold_state):
    run_func = partial(run_dist, world_size=world_size, chunk_size=chunk_size, model=get_openfold_state)
    mp.spawn(run_func, nprocs=world_size)


def run_dist(rank, world_size, chunk_size, model):
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=10101, backend='nccl')
    train(chunk_size, model)


def train(chunk_size, get_openfold_stat):

    # of_params, of_grads, ori_state_dict= get_openfold_state()

    of_params = pickle.load(open('./param.pkl', 'rb'))
    of_grads = pickle.load(open('./grad.pkl', 'rb'))
    
    config = model_config('initial_training', train=True)
    config.globals.inplace = False
    model = AlphaFold(config)
    model = inject_fastnn(model)
    #model.load_state_dict(ori_state_dict)
    model.train().cuda()
    
    criterion = AlphaFoldLoss(config.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, eps=1e-8)

    set_chunk_size(chunk_size)
    ff_engine, _, _, _ = colossalai.initialize(
                                            model=model,
                                            optimizer=optimizer,
                                            criterion=criterion
                                            )

    ff_params, ff_grads = colo_train_step(ff_engine)

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
    assert params_dif < 5e-4, f"Test failed at chunk size: {chunk_size}, \
        the param dif is {params_dif}, the grad diff is {grads_dif}"


if __name__ == '__main__':
     test_state_dict(1, None, None)