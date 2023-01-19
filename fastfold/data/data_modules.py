# Copyright 2022 HPC-AI Tech Inc.
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

from functools import partial
import json
import logging
import os
from typing import Optional, Sequence, List, Any

import ml_collections as mlc
import torch
from colossalai.utils import is_using_ddp
from fastfold.data import (
    data_pipeline,
    feature_pipeline,
    mmcif_parsing,
    templates,
)
from fastfold.utils.tensor_utils import tensor_tree_map, dict_multimap


class OpenFoldSingleDataset(torch.utils.data.Dataset):
    def __init__(self,
        data_dir: str,
        alignment_dir: str, 
        template_mmcif_dir: str,
        max_template_date: str,
        config: mlc.ConfigDict,
        kalign_binary_path: str = '/usr/bin/kalign',
        max_template_hits: int = 4,
        obsolete_pdbs_file_path: Optional[str] = None,
        template_release_dates_cache_path: Optional[str] = None,
        shuffle_top_k_prefiltered: Optional[int] = None,
        treat_pdb_as_distillation: bool = True,
        mapping_path: Optional[str] = None,
        mode: str = "train", 
        _output_raw: bool = False,
        _alignment_index: Optional[Any] = None
    ):
        """
            Args:
                data_dir:
                    A path to a directory containing mmCIF files (in train
                    mode) or FASTA files (in inference mode).
                alignment_dir:
                    A path to a directory containing only data in the format 
                    output by an AlignmentRunner 
                    (defined in openfold.features.alignment_runner).
                    I.e. a directory of directories named {PDB_ID}_{CHAIN_ID}
                    or simply {PDB_ID}, each containing .a3m, .sto, and .hhr
                    files.
                template_mmcif_dir:
                    Path to a directory containing template mmCIF files.
                config:
                    A dataset config object. See openfold.config
                kalign_binary_path:
                    Path to kalign binary.
                max_template_hits:
                    An upper bound on how many templates are considered. During
                    training, the templates ultimately used are subsampled
                    from this total quantity.
                template_release_dates_cache_path:
                    Path to the output of scripts/generate_mmcif_cache.
                obsolete_pdbs_file_path:
                    Path to the file containing replacements for obsolete PDBs.
                shuffle_top_k_prefiltered:
                    Whether to uniformly shuffle the top k template hits before
                    parsing max_template_hits of them. Can be used to
                    approximate DeepMind's training-time template subsampling
                    scheme much more performantly.
                treat_pdb_as_distillation:
                    Whether to assume that .pdb files in the data_dir are from
                    the self-distillation set (and should be subjected to
                    special distillation set preprocessing steps).
                mode:
                    "train", "val", or "predict"
        """
        super(OpenFoldSingleDataset, self).__init__()
        self.data_dir = data_dir
        self.alignment_dir = alignment_dir
        self.config = config
        self.treat_pdb_as_distillation = treat_pdb_as_distillation
        self.mode = mode
        self._output_raw = _output_raw
        self._alignment_index = _alignment_index

        valid_modes = ["train", "eval", "predict"]
        if(mode not in valid_modes):
            raise ValueError(f'mode must be one of {valid_modes}')

        if(template_release_dates_cache_path is None):
            logging.warning(
                "Template release dates cache does not exist. Remember to run "
                "scripts/generate_mmcif_cache.py before running OpenFold"
            )

        if(_alignment_index is not None):
            self._chain_ids = list(_alignment_index.keys())
        elif(mapping_path is None):
            self._chain_ids = list(os.listdir(alignment_dir))
        else:
            with open(mapping_path, "r") as f:
                self._chain_ids = [l.strip() for l in f.readlines()]
        
        self._chain_id_to_idx_dict = {
            chain: i for i, chain in enumerate(self._chain_ids)
        }

        template_featurizer = templates.TemplateHitFeaturizer(
            mmcif_dir=template_mmcif_dir,
            max_template_date=max_template_date,
            max_hits=max_template_hits,
            kalign_binary_path=kalign_binary_path,
            release_dates_path=template_release_dates_cache_path,
            obsolete_pdbs_path=obsolete_pdbs_file_path,
            _shuffle_top_k_prefiltered=shuffle_top_k_prefiltered,
        )

        self.data_pipeline = data_pipeline.DataPipeline(
            template_featurizer=template_featurizer,
        )

        if(not self._output_raw):
            self.feature_pipeline = feature_pipeline.FeaturePipeline(config) 

    def _parse_mmcif(self, path, file_id, chain_id, alignment_dir, _alignment_index):
        with open(path, 'r') as f:
            mmcif_string = f.read()

        mmcif_object = mmcif_parsing.parse(
            file_id=file_id, mmcif_string=mmcif_string
        )

        # Crash if an error is encountered. Any parsing errors should have
        # been dealt with at the alignment stage.
        if(mmcif_object.mmcif_object is None):
            raise list(mmcif_object.errors.values())[0]

        mmcif_object = mmcif_object.mmcif_object

        data = self.data_pipeline.process_mmcif(
            mmcif=mmcif_object,
            alignment_dir=alignment_dir,
            chain_id=chain_id,
            _alignment_index=_alignment_index
        )

        return data

    def chain_id_to_idx(self, chain_id):
        return self._chain_id_to_idx_dict[chain_id]

    def idx_to_chain_id(self, idx):
        return self._chain_ids[idx]

    def __getitem__(self, idx):
        name = self.idx_to_chain_id(idx)
        alignment_dir = os.path.join(self.alignment_dir, name)

        _alignment_index = None
        if(self._alignment_index is not None):
            alignment_dir = self.alignment_dir
            _alignment_index = self._alignment_index[name]

        if(self.mode == 'train' or self.mode == 'eval'):
            spl = name.rsplit('_', 1)
            if(len(spl) == 2):
                file_id, chain_id = spl
            else:
                file_id, = spl
                chain_id = None

            path = os.path.join(self.data_dir, file_id)
            if(os.path.exists(path + ".cif")):
                data = self._parse_mmcif(
                    path + ".cif", file_id, chain_id, alignment_dir, _alignment_index,
                )
            elif(os.path.exists(path + ".core")):
                data = self.data_pipeline.process_core(
                    path + ".core", alignment_dir, _alignment_index,
                )
            elif(os.path.exists(path + ".pdb")):
                data = self.data_pipeline.process_pdb(
                    pdb_path=path + ".pdb",
                    alignment_dir=alignment_dir,
                    is_distillation=self.treat_pdb_as_distillation,
                    chain_id=chain_id,
                    _alignment_index=_alignment_index,
                )
            else:
                raise ValueError("Invalid file type")
        else:
            path = os.path.join(name, name + ".fasta")
            data = self.data_pipeline.process_fasta(
                fasta_path=path,
                alignment_dir=alignment_dir,
                _alignment_index=_alignment_index,
            )

        if(self._output_raw):
            return data

        feats = self.feature_pipeline.process_features(
            data, self.mode
        )

        return feats

    def __len__(self):
        return len(self._chain_ids) 


def deterministic_train_filter(
    chain_data_cache_entry: Any,
    max_resolution: float = 9.,
    max_single_aa_prop: float = 0.8,
) -> bool:
    # Hard filters
    resolution = chain_data_cache_entry.get("resolution", None)
    if(resolution is not None and resolution > max_resolution):
        return False

    seq = chain_data_cache_entry["seq"]
    counts = {}
    for aa in seq:
        counts.setdefault(aa, 0)
        counts[aa] += 1
    largest_aa_count = max(counts.values())
    largest_single_aa_prop = largest_aa_count / len(seq)
    if(largest_single_aa_prop > max_single_aa_prop):
        return False

    return True


def get_stochastic_train_filter_prob(
    chain_data_cache_entry: Any,
) -> List[float]:
    # Stochastic filters
    probabilities = []
    
    cluster_size = chain_data_cache_entry.get("cluster_size", None)
    if(cluster_size is not None and cluster_size > 0):
        probabilities.append(1 / cluster_size)
    
    chain_length = len(chain_data_cache_entry["seq"])
    probabilities.append((1 / 512) * (max(min(chain_length, 512), 256)))

    # Risk of underflow here?
    out = 1
    for p in probabilities:
        out *= p

    return out


class OpenFoldDataset(torch.utils.data.Dataset):
    """
        Implements the stochastic filters applied during AlphaFold's training.
        Because samples are selected from constituent datasets randomly, the
        length of an OpenFoldFilteredDataset is arbitrary. Samples are selected
        and filtered once at initialization.
    """
    def __init__(self,
        datasets: Sequence[OpenFoldSingleDataset],
        probabilities: Sequence[int],
        epoch_len: int,
        chain_data_cache_paths: List[str],
        generator: torch.Generator = None,
        _roll_at_init: bool = True,
    ):
        self.datasets = datasets
        self.probabilities = probabilities
        self.epoch_len = epoch_len
        self.generator = generator
        
        self.chain_data_caches = []
        for path in chain_data_cache_paths:
            with open(path, "r") as fp:
                self.chain_data_caches.append(json.load(fp))

        def looped_shuffled_dataset_idx(dataset_len):
            while True:
                # Uniformly shuffle each dataset's indices
                weights = [1. for _ in range(dataset_len)]
                shuf = torch.multinomial(
                    torch.tensor(weights),
                    num_samples=dataset_len,
                    replacement=False,
                    generator=self.generator,
                )
                for idx in shuf:
                    yield idx

        def looped_samples(dataset_idx):
            max_cache_len = int(epoch_len * probabilities[dataset_idx])
            dataset = self.datasets[dataset_idx]
            idx_iter = looped_shuffled_dataset_idx(len(dataset))
            chain_data_cache = self.chain_data_caches[dataset_idx]
            while True:
                weights = []
                idx = []
                for _ in range(max_cache_len):
                    candidate_idx = next(idx_iter)
                    chain_id = dataset.idx_to_chain_id(candidate_idx)
                    chain_data_cache_entry = chain_data_cache[chain_id]
                    if(not deterministic_train_filter(chain_data_cache_entry)):
                        continue

                    p = get_stochastic_train_filter_prob(
                        chain_data_cache_entry,
                    )
                    weights.append([1. - p, p])
                    idx.append(candidate_idx)

                samples = torch.multinomial(
                    torch.tensor(weights),
                    num_samples=1,
                    generator=self.generator,
                )
                samples = samples.squeeze()

                cache = [i for i, s in zip(idx, samples) if s]

                for datapoint_idx in cache:
                    yield datapoint_idx

        self._samples = [looped_samples(i) for i in range(len(self.datasets))]

        if(_roll_at_init):
            self.reroll()

    def __getitem__(self, idx):
        dataset_idx, datapoint_idx = self.datapoints[idx]
        return self.datasets[dataset_idx][datapoint_idx]

    def __len__(self):
        return self.epoch_len

    def reroll(self):
        dataset_choices = torch.multinomial(
            torch.tensor(self.probabilities),
            num_samples=self.epoch_len,
            replacement=True,
            generator=self.generator,
        )

        self.datapoints = []
        for dataset_idx in dataset_choices:
            samples = self._samples[dataset_idx]
            datapoint_idx = next(samples)
            self.datapoints.append((dataset_idx, datapoint_idx))


class OpenFoldBatchCollator:
    def __init__(self, config, stage="train"):
        self.stage = stage
        self.feature_pipeline = feature_pipeline.FeaturePipeline(config)

    def __call__(self, raw_prots):
        processed_prots = []
        for prot in raw_prots:
            features = self.feature_pipeline.process_features(
                prot, self.stage
            )
            processed_prots.append(features)
        # By this stack, the batch dimension is processed and added.
        stack_fn = partial(torch.stack, dim=0)
        # I have modified some codes. Now if the bs=1, the shape will be [...] rather than [1, ...]
        # If bs>1(not allowed), the shape would be still [2, ...]
        return dict_multimap(stack_fn, processed_prots) 


class OpenFoldDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, config, stage="train", generator=None, **kwargs):
        super().__init__(dataset, **kwargs)
        self.config = config
        self.stage = stage    

        if(generator is None):
            generator = torch.Generator()
        
        self.generator = generator
        self._prep_batch_properties_probs()

    def _prep_batch_properties_probs(self):
        keyed_probs = []
        stage_cfg = self.config[self.stage]

        max_iters = self.config.common.max_recycling_iters
        if(stage_cfg.supervised):
            clamp_prob = self.config.supervised.clamp_prob
            keyed_probs.append(
                ("use_clamped_fape", [1 - clamp_prob, clamp_prob])
            )
        
        if(stage_cfg.uniform_recycling):
            recycling_probs = [
                1. / (max_iters + 1) for _ in range(max_iters + 1)
            ]
        else:
            recycling_probs = [
                0. for _ in range(max_iters + 1)
            ]
            recycling_probs[-1] = 1.
        
        keyed_probs.append(
            ("no_recycling_iters", recycling_probs)
        )

        keys, probs = zip(*keyed_probs)
        max_len = max([len(p) for p in probs])
        padding = [[0.] * (max_len - len(p)) for p in probs] 
        
        self.prop_keys = keys
        self.prop_probs_tensor = torch.tensor(
            [p + pad for p, pad in zip(probs, padding)],
            dtype=torch.float32,
        )

    def _add_batch_properties(self, batch):
        samples = torch.multinomial(
            self.prop_probs_tensor,
            num_samples=1, # 1 per row
            replacement=True,
            generator=self.generator
        )

        aatype = batch["aatype"]
        batch_dims = aatype.shape[:-2]
        recycling_dim = aatype.shape[-1]
        no_recycling = recycling_dim
        for i, key in enumerate(self.prop_keys):
            sample = int(samples[i][0])
            sample_tensor = torch.tensor(
                sample, 
                device=aatype.device, 
                requires_grad=False
            )
            orig_shape = sample_tensor.shape
            sample_tensor = sample_tensor.view(
                (1,) * len(batch_dims) + sample_tensor.shape + (1,)
            )
            sample_tensor = sample_tensor.expand(
                batch_dims + orig_shape + (recycling_dim,)
            )
            batch[key] = sample_tensor

            if(key == "no_recycling_iters"):
                no_recycling = sample 
        
        resample_recycling = lambda t: t[..., :no_recycling + 1]
        batch = tensor_tree_map(resample_recycling, batch)

        return batch

    def __iter__(self):
        it = super().__iter__()

        def _batch_prop_gen(iterator):
            for batch in iterator:
                yield self._add_batch_properties(batch)

        return _batch_prop_gen(it)


def SetupTrainDataset(
    config: mlc.ConfigDict,
    template_mmcif_dir: str,
    max_template_date: str,
    train_data_dir: Optional[str] = None,
    train_alignment_dir: Optional[str] = None,
    train_chain_data_cache_path: Optional[str] = None,
    distillation_data_dir: Optional[str] = None,
    distillation_alignment_dir: Optional[str] = None,
    distillation_chain_data_cache_path: Optional[str] = None,
    val_data_dir: Optional[str] = None,
    val_alignment_dir: Optional[str] = None,
    kalign_binary_path: str = '/usr/bin/kalign',
    train_mapping_path: Optional[str] = None,
    distillation_mapping_path: Optional[str] = None,
    obsolete_pdbs_file_path: Optional[str] = None,
    template_release_dates_cache_path: Optional[str] = None,
    train_epoch_len: int = 50000, 
    _alignment_index_path: Optional[str] = None,
    **kwargs,
):

    if(train_data_dir is None or train_alignment_dir is None):
        raise ValueError(
            'train_data_dir and train_alignment_dir must be specified'
        )     
    elif(val_data_dir is not None and val_alignment_dir is None):
        raise ValueError(
            'If val_data_dir is specified, val_alignment_dir must '
            'be specified as well'
    )

    _alignment_index = None
    if(_alignment_index_path is not None):
        with open(_alignment_index_path, "r") as fp:
            _alignment_index = json.load(fp)

    dataset_gen = partial(OpenFoldSingleDataset,
            template_mmcif_dir=template_mmcif_dir,
            max_template_date=max_template_date,
            config=config,
            kalign_binary_path=kalign_binary_path,
            template_release_dates_cache_path=
                template_release_dates_cache_path,
            obsolete_pdbs_file_path=
                obsolete_pdbs_file_path,
        )

    train_dataset = dataset_gen(
        data_dir=train_data_dir,
        alignment_dir=train_alignment_dir,
        mapping_path=train_mapping_path,
        max_template_hits=config.train.max_template_hits,
        shuffle_top_k_prefiltered=
            config.train.shuffle_top_k_prefiltered,
        treat_pdb_as_distillation=False,
        mode="train",
        _output_raw=True,
        _alignment_index=_alignment_index,
    )

    distillation_dataset = None
    if(distillation_data_dir is not None):
        distillation_dataset = dataset_gen(
            data_dir=distillation_data_dir,
            alignment_dir=distillation_alignment_dir,
            mapping_path=distillation_mapping_path,
            max_template_hits=config.train.max_template_hits,
            treat_pdb_as_distillation=True,
            mode="train",
            _output_raw=True,
        )

        d_prob = config.train.distillation_prob
    
    if(distillation_dataset is not None):
        datasets = [train_dataset, distillation_dataset]
        d_prob = config.train.distillation_prob
        probabilities = [1 - d_prob, d_prob]
        chain_data_cache_paths = [
            train_chain_data_cache_path,
            distillation_chain_data_cache_path,
        ]
    else:
        datasets = [train_dataset]
        probabilities = [1.]   
        chain_data_cache_paths = [
            train_chain_data_cache_path,
        ]

    train_dataset = OpenFoldDataset(
        datasets=datasets,
        probabilities=probabilities,
        epoch_len=train_epoch_len,
        chain_data_cache_paths=chain_data_cache_paths,
        _roll_at_init=False,
    )

    if(val_data_dir is not None):
        eval_dataset = dataset_gen(
            data_dir=val_data_dir,
            alignment_dir=val_alignment_dir,
            mapping_path=None,
            max_template_hits=config.eval.max_template_hits,
            mode="eval",
            _output_raw=True,
        )
    else:
        eval_dataset = None
    
    return train_dataset, eval_dataset


def TrainDataLoader(
    config: mlc.ConfigDict,
    train_dataset: torch.utils.data.Dataset,
    test_dataset: Optional[torch.utils.data.Dataset] = None,
    batch_seed: Optional[int] = None,
):

    if not config.data_module.data_loaders.batch_size == 1:
        raise ValueError("Only support batch size equals to 1")

    generator = torch.Generator()
    if(batch_seed is not None):
        generator = generator.manual_seed(batch_seed)

    train_batch_collator = OpenFoldBatchCollator(config, "train")
    train_sampler = None
    if is_using_ddp():
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataset.reroll()
    train_dataloader = OpenFoldDataLoader(
        dataset=train_dataset,
        config=config,
        stage="train",
        generator=generator,
        batch_size=config.data_module.data_loaders.batch_size,
        num_workers=config.data_module.data_loaders.num_workers,
        collate_fn=train_batch_collator,
        sampler=train_sampler,
    )

    test_dataloader = None
    if test_dataset is not None:
        test_sampler = None
        if is_using_ddp():
            test_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_batch_collator = OpenFoldBatchCollator(config, "test")
        test_dataloader = OpenFoldDataLoader(
            dataset=test_dataset,
            config=config,
            stage="test",
            generator=generator,
            batch_size=config.data_module.data_loaders.batch_size,
            num_workers=config.data_module.data_loaders.num_workers,
            collate_fn=test_batch_collator,
            sampler=test_sampler,
        )

    return train_dataloader, test_dataloader
