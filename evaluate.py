from functools import partial
import argparse
import time
from datetime import date

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

import fastfold
from fastfold.config import model_config
from fastfold.model.hub import AlphaFold
from fastfold.model.fastnn import set_chunk_size
from fastfold.utils.inject_fastnn import inject_fastnn
from fastfold.utils.import_weights import import_jax_weights_
from fastfold.data import data_pipeline, feature_pipeline, templates
from fastfold.data.data_modules import OpenFoldSingleDataset, OpenFoldBatchCollator, OpenFoldDataLoader
from fastfold.utils.validation_utils import compute_validation_metrics
from fastfold.utils.tensor_utils import tensor_tree_map

if int(torch.__version__.split(".")[0]) >= 1 and int(torch.__version__.split(".")[1]) > 11:
    torch.backends.cuda.matmul.allow_tf32 = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cameo_path", type=str)
    parser.add_argument("--alignment_dir", type=str)
    parser.add_argument("--model_name", type=str, default="model_1")
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--template_mmcif_dir", type=str)
    parser.add_argument(
        '--max_template_date',
        type=str,
        default=date.today().strftime("%Y-%m-%d"),
    )
    parser.add_argument('--release_dates_path', type=str, default=None)
    parser.add_argument('--obsolete_pdbs_path', type=str, default=None)
    parser.add_argument('--kalign_binary_path', type=str,
                        default='kalign')
    args = parser.parse_args()
    config = model_config(args.model_name)

    dataset_gen = partial(OpenFoldSingleDataset,
        template_mmcif_dir=args.template_mmcif_dir,
        max_template_date=args.max_template_date,
        config=config.data,
        kalign_binary_path=args.kalign_binary_path
    )

    eval_dataset = dataset_gen(
        data_dir=args.cameo_path,
        alignment_dir=args.alignment_dir,
        mapping_path=None,
        max_template_hits=4,
        mode="eval",
        _output_raw=True,
    )
    test_batch_collator = OpenFoldBatchCollator(config.data, "eval")
    generator = torch.Generator()
    test_dataloader = OpenFoldDataLoader(
            dataset=eval_dataset,
            config=config.data,
            stage="eval",
            generator=generator,
            batch_size=config.data.data_module.data_loaders.batch_size,
            num_workers=config.data.data_module.data_loaders.num_workers,
            collate_fn=test_batch_collator, 
    ) 

    fastfold.distributed.init_dap()

    config = model_config(args.model_name)
    model = AlphaFold(config)

    import_jax_weights_(model, "./data/params/params_model_1.npz", version=args.model_name)

    model = inject_fastnn(model)
    model = model.eval()
    model = model.cuda()

    model = DDP(model)

    if args.ckpt_path:
        print(f">>>> Load {args.ckpt_path}")
        checkpoint = torch.load(args.ckpt_path)
        model.load_state_dict(checkpoint['model'])

    config.globals.chunk_size = None
    config.globals.inplace = False

    with torch.no_grad():

        idx = 0
        lddt_ca = []
        for batch in test_dataloader:
            batch = {k: torch.as_tensor(v).cuda() for k, v in batch.items()}

            print(batch["seq_length"][0].item())
            if batch["seq_length"][0].item() > 512:
                config.globals.chunk_size = 512
                set_chunk_size(model.module.globals.chunk_size)

            t = time.perf_counter()
            out = model(batch)
            print(f"Inference time: {time.perf_counter() - t}")

            batch = tensor_tree_map(lambda t: t[..., -1], batch)
            other_metrics = compute_validation_metrics(
                batch, 
                out,
                superimposition_metrics=True
            ) 
            print(other_metrics)
            lddt_ca.append(other_metrics['lddt_ca'].item())

            config.globals.chunk_size = None
            set_chunk_size(model.module.globals.chunk_size)

            idx += 1

            if idx == 20:
                break

    lddt_ca = np.array(lddt_ca)
    print(f"lddt_ca: {np.mean(lddt_ca)}, {np.median(lddt_ca)}")

if __name__ == '__main__':
    main()

