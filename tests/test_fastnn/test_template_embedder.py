import torch
import pytest
import pickle
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
from fastfold.utils.tensor_utils import tensor_tree_map
from fastfold.utils.test_utils import get_param_path, get_data_path


@pytest.fixture(scope="module")
def get_openfold_module_and_data():
    with torch.no_grad():
        config = model_config('model_1')
        config.globals.inplace = False
        target_module = AlphaFold(config)
        import_jax_weights_(target_module, get_param_path())
        
        fast_module = copy.deepcopy(target_module)
        fast_module = inject_fastnn(fast_module)
        fast_module = fast_module.template_embedder
        fast_module = fast_module.eval().cuda()
        
        target_module = target_module.template_embedder
        target_module = target_module.eval().cuda()
        
        batch = pickle.load(open(get_data_path(), 'rb'))
        fetch_cur_batch = lambda t: t[..., 0]
        feats = tensor_tree_map(fetch_cur_batch, batch)
        feats = {k: v.cuda() for k, v in feats.items() if k.startswith("template_")}

        seq_len = 33
        z = torch.randn((seq_len, seq_len, 128)).cuda()
        z_mask = torch.ones((seq_len, seq_len)).cuda().to(dtype=z.dtype)
        
        template_embeds = target_module(copy.deepcopy(feats), z, z_mask.to(dtype=z.dtype), 0, None)
        z_out = z + template_embeds["template_pair_embedding"]
    return fast_module, z_out, feats, z, z_mask


@pytest.mark.parametrize('world_size', [1, 2])
@pytest.mark.parametrize('chunk_size', [None, 4]) # should set 4 to test offload
@pytest.mark.parametrize('inplace', [False, True])
def test_state_dict(world_size, chunk_size, inplace, get_openfold_module_and_data): 
    run_func = partial(_test_template_embedder, world_size=world_size, chunk_size=chunk_size, 
                       inplace=inplace, get_openfold_module_and_data=get_openfold_module_and_data)
    mp.spawn(run_func, nprocs=world_size)


def _test_template_embedder(rank, world_size, chunk_size, inplace, get_openfold_module_and_data):
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    # init distributed for Dynamic Axial Parallelism
    fastfold.distributed.init_dap()    

    fast_module, z_out, feats, z, z_mask = get_openfold_module_and_data
    
    fast_module = copy.deepcopy(fast_module).cuda()
    template_feats = copy.deepcopy(feats)
    for k, v in template_feats.items():
        template_feats[k] = v.cuda()

    with torch.no_grad():
        set_chunk_size(chunk_size)
        if inplace:
            template_embeds = fast_module(copy.deepcopy(template_feats), copy.deepcopy(z).cuda(), z_mask.to(dtype=z.dtype).cuda(), 0, chunk_size, inplace=inplace)
            z_fast = template_embeds["template_pair_embedding"]
        else:
            template_embeds = fast_module(copy.deepcopy(template_feats), copy.deepcopy(z).cuda(), z_mask.to(dtype=z.dtype).cuda(), 0, chunk_size)
            z_fast = z.cuda() + template_embeds["template_pair_embedding"]

    error = torch.mean(torch.abs(z_out.cuda() - z_fast))
    assert error < 5e-4, f"Test z failed at chunk size: {chunk_size}, inplace: {inplace}. The position dif is {error}"
