import os
import random
import torch
import numpy as np
import colossalai
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer import HybridAdam

from fastfold.config import model_config
from fastfold.model.hub import AlphaFold, AlphaFoldLRScheduler, AlphaFoldLoss
from fastfold.utils.inject_fastnn import inject_fastnn
from fastfold.data.data_modules import SetupTrainDataset, TrainDataLoader
from fastfold.utils.tensor_utils import tensor_tree_map
from fastfold.utils.validation_utils import compute_validation_metrics
#import logging
#logging.disable(logging.WARNING)
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def log_loss(loss_breakdown, batch, outputs, train=True):
    loss_info = ''
    for loss_name, loss_value in loss_breakdown.items():
        loss_info += (f' {loss_name}=' + "{:.3f}".format(loss_value))
    with torch.no_grad():
        other_metrics = compute_validation_metrics(
            batch, 
            outputs,
            superimposition_metrics=(not train)
        )
    for loss_name, loss_value in other_metrics.items():
        loss_info += (f' {loss_name}=' + "{:.3f}".format(loss_value))
    return loss_info


def main():
    parser = colossalai.get_default_parser()
    parser.add_argument('--from_torch', default=False, action='store_true')
    parser.add_argument(
        "--template_mmcif_dir", type=str,
        help="Directory containing mmCIF files to search for templates"
    )
    parser.add_argument(
        "--max_template_date", type=str,
        help='''Cutoff for all templates. In training mode, templates are also 
                filtered by the release date of the target'''
    )
    parser.add_argument(
        "--train_data_dir", type=str,
        help="Directory containing training mmCIF files"
    )
    parser.add_argument(
        "--train_alignment_dir", type=str,
        help="Directory containing precomputed training alignments"
    )
    parser.add_argument(
        "--train_chain_data_cache_path", type=str, default=None,
    )
    parser.add_argument(
        "--distillation_data_dir", type=str, default=None,
        help="Directory containing training PDB files"
    )
    parser.add_argument(
        "--distillation_alignment_dir", type=str, default=None,
        help="Directory containing precomputed distillation alignments"
    )
    parser.add_argument(
        "--distillation_chain_data_cache_path", type=str, default=None,
    )
    parser.add_argument(
        "--val_data_dir", type=str, default=None,
        help="Directory containing validation mmCIF files"
    )
    parser.add_argument(
        "--val_alignment_dir", type=str, default=None,
        help="Directory containing precomputed validation alignments"
    )
    parser.add_argument(
        "--kalign_binary_path", type=str, default='/usr/bin/kalign',
        help="Path to the kalign binary"
    )
    parser.add_argument(
        "--train_filter_path", type=str, default=None,
        help='''Optional path to a text file containing names of training
                examples to include, one per line. Used to filter the training 
                set'''
    )
    parser.add_argument(
        "--distillation_filter_path", type=str, default=None,
        help="""See --train_filter_path"""
    )
    parser.add_argument(
        "--obsolete_pdbs_file_path", type=str, default=None,
        help="""Path to obsolete.dat file containing list of obsolete PDBs and 
             their replacements."""
    )
    parser.add_argument(
        "--template_release_dates_cache_path", type=str, default=None,
        help="""Output of scripts/generate_mmcif_cache.py run on template mmCIF
                files."""
    )
    parser.add_argument(
        "--train_epoch_len", type=int, default=10000,
        help=(
            "The virtual length of each training epoch. Stochastic filtering "
            "of training data means that training datasets have no "
            "well-defined length. This virtual length affects frequency of "
            "validation & checkpointing (by default, one of each per epoch)."
        )
    )
    parser.add_argument(
        "--_alignment_index_path", type=str, default=None,
        help="Training alignment index. See the README for instructions."
    )
    parser.add_argument(
        "--config_preset", type=str, default="initial_training",
        help=(
            'Config setting. Choose e.g. "initial_training", "finetuning", '
            '"model_1", etc. By default, the actual values in the config are '
            'used.'
        )
    )
    parser.add_argument(
        "--_distillation_structure_index_path", type=str, default=None,
    )
    parser.add_argument(
        "--distillation_alignment_index_path", type=str, default=None,
        help="Distillation alignment index. See the README for instructions."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=10000,
        help="The Max epochs of train"
    )
    parser.add_argument(
        "--log_interval", type=int, default=1,
        help="The interval steps of logging during training"
    )
    parser.add_argument(
        "--log_path", type=str, default='train_log',
        help="The path of log folder"
    )
    parser.add_argument(
        "--save_ckpt_path", type=str, default=None,
        help="The path where to save checkpoint, None means not save"
    )
    parser.add_argument(
        "--save_ckpt_interval", type=int, default=1,
        help="The interval epochs of save checkpoint"
    )
    parser.add_argument(
        "--dap_size", type=int, default=1,
        help="DAP size, recommended as 1 - nproc_per_node"
    )

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.from_torch:
        colossalai.launch_from_torch(config=dict(parallel=dict(tensor=dict(size=args.dap_size)), 
                                                torch_ddp=dict(static_graph=True)))
    disable_existing_loggers()
    logger = get_dist_logger()
    logger.log_to_file(args.log_path)

    config = model_config(args.config_preset, train=True)
    config.globals.inplace = False
    model = AlphaFold(config)
    model = inject_fastnn(model)


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

    optimizer = HybridAdam(model.parameters(), lr=1e-3, eps=1e-8)

    lr_scheduler = AlphaFoldLRScheduler(optimizer)
    

    engine, train_dataloader, test_dataloader, lr_scheduler = colossalai.initialize(
                                                                model=model,
                                                                optimizer=optimizer,
                                                                criterion=criterion,
                                                                lr_scheduler=lr_scheduler,
                                                                train_dataloader=train_dataloader,
                                                                test_dataloader=test_dataloader,
                                                                )
    

    logger.info('Start training.', ranks=[0])
    for epoch in range(args.max_epochs):
        engine.train()
        for i, batch in enumerate(train_dataloader):
            batch = {k: torch.as_tensor(v).cuda() for k, v in batch.items()}
            output = engine(batch)
            batch = tensor_tree_map(lambda t: t[..., -1], batch)
            loss, loss_breakdown = engine.criterion(
                    output, batch, _return_breakdown=True)
            if (i+1) % args.log_interval == 0:
                logger.info(f'Training, Epoch: {epoch}, Step: {i+1}, Global_Step: {epoch*len(train_dataloader)+i+1},' +
                            f' Loss:{log_loss(loss_breakdown, batch, output)}', ranks=[0])
            engine.zero_grad()
            engine.backward(loss)
            engine.step()
        lr_scheduler.step()
        
        if test_dataloader is not None:
            engine.eval()
            for i, batch in enumerate(test_dataloader):
                batch = {k: torch.as_tensor(v).cuda() for k, v in batch.items()}
                with torch.no_grad():
                    output = engine(batch)
                    batch = tensor_tree_map(lambda t: t[..., -1], batch)
                    batch["use_clamped_fape"] = 0.
                    _, loss_breakdown = engine.criterion(
                            output, batch, _return_breakdown=True)
                    logger.info(f'Validation, Step: {i+1}, \
                                Loss:{log_loss(loss_breakdown, batch, output, False)}', ranks=[0])
        
        if (args.save_ckpt_path is not None) and ( (epoch+1) % args.save_ckpt_interval == 0):
            torch.save(engine.model, os.path.join(args.save_ckpt_path, 'model.pth')) 
        

if __name__ == "__main__":
    main()
