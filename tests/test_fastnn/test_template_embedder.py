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


@pytest.mark.parametrize('world_size', [1, 2])
@pytest.mark.parametrize('chunk_size', [None, 3])
@pytest.mark.parametrize('inplace', [False, True])
def test_state_dict(world_size, chunk_size, inplace):
    config = model_config('model_1')
    config.globals.chunk_size = chunk_size
    config.globals.inplace = False
    target_module = AlphaFold(config)
    import_jax_weights_(target_module, get_param_path())

    fast_module = copy.deepcopy(target_module)
    fast_module = inject_fastnn(fast_module)

    target_module = target_module.template_embedder
    fast_module = fast_module.template_embedder
    target_module = target_module.eval()
    fast_module = fast_module.eval()
    
    batch = pickle.load(open(get_data_path(), 'rb'))
    fetch_cur_batch = lambda t: t[..., 0]
    feats = tensor_tree_map(fetch_cur_batch, batch)
    template_feats = {k: v for k, v in feats.items() if k.startswith("template_")}
    
    run_func = partial(_test_template_embedder, world_size=world_size, chunk_size=chunk_size, 
                       inplace=inplace, fast_module=fast_module, target_module=target_module, config=config, 
                       template_feats=template_feats)
    mp.spawn(run_func, nprocs=world_size)


def _test_template_embedder(rank, world_size, chunk_size, inplace, fast_module, target_module, config, template_feats):
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    # init distributed for Dynamic Axial Parallelism
    fastfold.distributed.init_dap()    
    
    target_module = target_module.cuda()
    fast_module = fast_module.cuda()

    template_feats = copy.deepcopy(template_feats)
    for k, v in template_feats.items():
        template_feats[k] = v.cuda()

    seq_len = 33
    z = torch.randn((seq_len, seq_len, 128)).cuda()
    z_mask = torch.ones((seq_len, seq_len)).cuda().to(dtype=z.dtype)

    with torch.no_grad():
        template_embeds = target_module(copy.deepcopy(template_feats), z, z_mask.to(dtype=z.dtype), 0, None)
        z_out = z + template_embeds["template_pair_embedding"]

        set_chunk_size(chunk_size)
        if inplace:
            template_embeds = fast_module(copy.deepcopy(template_feats), z, z_mask.to(dtype=z.dtype), 0, chunk_size, inplace=inplace)
            z_fast = template_embeds["template_pair_embedding"]
        else:
            template_embeds = fast_module(copy.deepcopy(template_feats), z, z_mask.to(dtype=z.dtype), 0, chunk_size)
            z_fast = z + template_embeds["template_pair_embedding"]

    error = torch.mean(torch.abs(z_out - z_fast))
    assert error < 1e-5, f"Test z failed at chunk size: {chunk_size}, inplace: {inplace}. The position dif is {error}"
