import argparse
import logging
import random

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

import fastfold.habana as habana
from fastfold.config import model_config
from fastfold.data.data_modules import SetupTrainDataset, TrainDataLoader
from fastfold.habana.distributed import init_dist, get_data_parallel_world_size
from fastfold.habana.inject_habana import inject_habana
from fastfold.model.hub import AlphaFold, AlphaFoldLoss, AlphaFoldLRScheduler
from fastfold.utils.tensor_utils import tensor_tree_map

import habana_frameworks.torch.core as htcore
from habana_frameworks.torch.hpex import hmp

logging.disable(logging.WARNING)

torch.multiprocessing.set_sharing_strategy('file_system')

from habana.hpuhelper import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_torch', default=False, action='store_true')
    parser.add_argument("--template_mmcif_dir",
                        type=str,
                        help="Directory containing mmCIF files to search for templates")
    parser.add_argument("--max_template_date",
                        type=str,
                        help='''Cutoff for all templates. In training mode, templates are also 
                filtered by the release date of the target''')
    parser.add_argument("--train_data_dir",
                        type=str,
                        help="Directory containing training mmCIF files")
    parser.add_argument("--train_alignment_dir",
                        type=str,
                        help="Directory containing precomputed training alignments")
    parser.add_argument(
        "--train_chain_data_cache_path",
        type=str,
        default=None,
    )
    parser.add_argument("--distillation_data_dir",
                        type=str,
                        default=None,
                        help="Directory containing training PDB files")
    parser.add_argument("--distillation_alignment_dir",
                        type=str,
                        default=None,
                        help="Directory containing precomputed distillation alignments")
    parser.add_argument(
        "--distillation_chain_data_cache_path",
        type=str,
        default=None,
    )
    parser.add_argument("--val_data_dir",
                        type=str,
                        default=None,
                        help="Directory containing validation mmCIF files")
    parser.add_argument("--val_alignment_dir",
                        type=str,
                        default=None,
                        help="Directory containing precomputed validation alignments")
    parser.add_argument("--kalign_binary_path",
                        type=str,
                        default='/usr/bin/kalign',
                        help="Path to the kalign binary")
    parser.add_argument("--train_filter_path",
                        type=str,
                        default=None,
                        help='''Optional path to a text file containing names of training
                examples to include, one per line. Used to filter the training 
                set''')
    parser.add_argument("--distillation_filter_path",
                        type=str,
                        default=None,
                        help="""See --train_filter_path""")
    parser.add_argument("--obsolete_pdbs_file_path",
                        type=str,
                        default=None,
                        help="""Path to obsolete.dat file containing list of obsolete PDBs and 
             their replacements.""")
    parser.add_argument("--template_release_dates_cache_path",
                        type=str,
                        default=None,
                        help="""Output of scripts/generate_mmcif_cache.py run on template mmCIF
                files.""")
    parser.add_argument("--train_epoch_len",
                        type=int,
                        default=10000,
                        help=("The virtual length of each training epoch. Stochastic filtering "
                              "of training data means that training datasets have no "
                              "well-defined length. This virtual length affects frequency of "
                              "validation & checkpointing (by default, one of each per epoch)."))
    parser.add_argument("--_alignment_index_path",
                        type=str,
                        default=None,
                        help="Training alignment index. See the README for instructions.")
    parser.add_argument("--config_preset",
                        type=str,
                        default="initial_training",
                        help=('Config setting. Choose e.g. "initial_training", "finetuning", '
                              '"model_1", etc. By default, the actual values in the config are '
                              'used.'))
    parser.add_argument(
        "--_distillation_structure_index_path",
        type=str,
        default=None,
    )
    parser.add_argument("--distillation_alignment_index_path",
                        type=str,
                        default=None,
                        help="Distillation alignment index. See the README for instructions.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # habana arguments
    parser.add_argument("--hmp",
                        action='store_true',
                        default=False,
                        help="Whether to use habana mixed precision")
    parser.add_argument("--hmp-bf16",
                        type=str,
                        default="./habana/ops_bf16.txt",
                        help="Path to bf16 ops list in hmp O1 mode")
    parser.add_argument("--hmp-fp32",
                        type=str,
                        default="./habana/ops_fp32.txt",
                        help="Path to fp32 ops list in hmp O1 mode")
    parser.add_argument("--hmp-opt-level",
                        type=str,
                        default='O1',
                        help="Choose optimization level for hmp")
    parser.add_argument("--hmp-verbose",
                        action='store_true',
                        default=False,
                        help='Enable verbose mode for hmp')

    args = parser.parse_args()

    habana.enable_habana()

    init_dist()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = model_config(args.config_preset, train=True)
    config.globals.inplace = False
    model = AlphaFold(config)
    model = inject_habana(model)

    model = model.to(device="hpu")
    if get_data_parallel_world_size() > 1:
        model = DDP(model, gradient_as_bucket_view=True, bucket_cap_mb=400)

    train_dataset, test_dataset = SetupTrainDataset(
        config=config.data,
        template_mmcif_dir=args.template_mmcif_dir,
        max_template_date=args.max_template_date,
        train_data_dir=args.train_data_dir,
        train_alignment_dir=args.train_alignment_dir,
        train_chain_data_cache_path=args.train_chain_data_cache_path,
        distillation_data_dir=args.distillation_data_dir,
        distillation_alignment_dir=args.distillation_alignment_dir,
        distillation_chain_data_cache_path=args.distillation_chain_data_cache_path,
        val_data_dir=args.val_data_dir,
        val_alignment_dir=args.val_alignment_dir,
        kalign_binary_path=args.kalign_binary_path,
        # train_mapping_path=args.train_mapping_path,
        # distillation_mapping_path=args.distillation_mapping_path,
        obsolete_pdbs_file_path=args.obsolete_pdbs_file_path,
        template_release_dates_cache_path=args.template_release_dates_cache_path,
        train_epoch_len=args.train_epoch_len,
        _alignment_index_path=args._alignment_index_path,
    )

    train_dataloader, test_dataloader = TrainDataLoader(
        config=config.data,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_seed=args.seed,
    )

    criterion = AlphaFoldLoss(config.loss)

    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-8)
    from habana_frameworks.torch.hpex.optimizers import FusedAdamW
    optimizer = FusedAdamW(model.parameters(), lr=1e-3, eps=1e-8)

    lr_scheduler = AlphaFoldLRScheduler(optimizer)

    if args.hmp:
        hmp.convert(opt_level='O1',
                    bf16_file_path=args.hmp_bf16,
                    fp32_file_path=args.hmp_fp32,
                    isVerbose=args.hmp_verbose)
        print("========= HMP ENABLED!!")

    idx = 0
    for epoch in range(200):
        model.train()
        train_dataloader = tqdm(train_dataloader)
        for batch in train_dataloader:
            perf = hpu_perf("train step")
            batch = {k: torch.as_tensor(v).to(device="hpu", non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad()
            perf.checknow("prepare input and zero grad")
            output = model(batch)
            perf.checknow("forward")
            
            batch = tensor_tree_map(lambda t: t[..., -1], batch)
            perf.checknow("prepare loss input")
            loss, loss_breakdown = criterion(output, batch, _return_breakdown=True)
            perf.checknow("loss")

            loss.backward()
            if idx % 10 == 0:
                train_dataloader.set_postfix(loss=float(loss))
            perf.checknow("backward")

            with hmp.disable_casts():
                optimizer.step()
            perf.checknow("optimizer")
            idx += 1

        lr_scheduler.step()

        if test_dataloader is not None:
            model.eval()
            train_dataloader = tqdm(train_dataloader)
            for batch in test_dataloader:
                batch = {k: torch.as_tensor(v).to(device="hpu") for k, v in batch.items()}
                with torch.no_grad():
                    output = model(batch)
                    batch = tensor_tree_map(lambda t: t[..., -1], batch)
                    _, loss_breakdown = criterion(output, batch, _return_breakdown=True)

                    htcore.mark_step()
                    train_dataloader.set_postfix(loss=float(loss))


if __name__ == "__main__":
    main()
