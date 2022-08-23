# Copyright 2021 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import torch
import ml_collections as mlc

import fastfold
from fastfold.model.hub import AlphaFold
from fastfold.config import model_config
from fastfold.model.fastnn import set_chunk_size
from fastfold.utils import inject_fastnn
from test_data_utils import random_extra_msa_feats, random_template_feats
from fastfold.data import data_transforms
from fastfold.utils.tensor_utils import tensor_tree_map


consts = mlc.ConfigDict(
    {
        "n_res": 11,
        "n_seq": 13,
        "n_templ": 3,
        "n_extra": 17,
    }
)

def inference():
    fastfold.distributed.init_dap()

    n_seq = consts.n_seq
    n_templ = consts.n_templ
    n_res = consts.n_res
    n_extra_seq = consts.n_extra


    config = model_config('model_1')
    model = AlphaFold(config)
    model = inject_fastnn(model)
    model.eval()
    model.cuda()

    set_chunk_size(model.globals.chunk_size)

    batch = {}
    tf = torch.randint(config.model.input_embedder.tf_dim - 1, size=(n_res,))
    batch["target_feat"] = torch.nn.functional.one_hot(
        tf, config.model.input_embedder.tf_dim).float()
    batch["aatype"] = torch.argmax(batch["target_feat"], dim=-1)
    batch["residue_index"] = torch.arange(n_res)
    batch["msa_feat"] = torch.rand((n_seq, n_res, config.model.input_embedder.msa_dim))
    t_feats = random_template_feats(n_templ, n_res)
    batch.update({k: torch.tensor(v) for k, v in t_feats.items()})
    extra_feats = random_extra_msa_feats(n_extra_seq, n_res)
    batch.update({k: torch.tensor(v) for k, v in extra_feats.items()})
    batch["msa_mask"] = torch.randint(low=0, high=2, size=(n_seq, n_res)).float()
    batch["seq_mask"] = torch.randint(low=0, high=2, size=(n_res,)).float()
    batch.update(data_transforms.make_atom14_masks(batch))
    batch["no_recycling_iters"] = torch.tensor(2.)
    add_recycling_dims = lambda t: (
            t.unsqueeze(-1).expand(*t.shape, config.data.common.max_recycling_iters))
    batch = tensor_tree_map(add_recycling_dims, batch)


    with torch.no_grad():
        batch = {k: torch.as_tensor(v).cuda() for k, v in batch.items()}
        t = time.perf_counter()
        out = model(batch)
        print(f"Inference time: {time.perf_counter() - t}")

if __name__ == "__main__":
    inference()
    print("Inference Test Passed!")