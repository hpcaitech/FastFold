# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
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

import argparse
import contextlib
import os
import pickle
import random
import shutil
import sys
import tempfile
import time
from datetime import date

import numpy as np
import torch
import torch.multiprocessing as mp

import habana_frameworks.torch.core as htcore

import fastfold.habana as habana
import fastfold.relax.relax as relax
from fastfold.common import protein, residue_constants
from fastfold.config import model_config
from fastfold.data import data_pipeline, feature_pipeline, templates
from fastfold.data.parsers import parse_fasta
from fastfold.habana.distributed import init_dist
from fastfold.habana.fastnn.ops import set_chunk_size
from fastfold.habana.inject_habana import inject_habana
from fastfold.model.hub import AlphaFold
from fastfold.model.nn.triangular_multiplicative_update import \
    set_fused_triangle_multiplication
from fastfold.utils.import_weights import import_jax_weights_
from fastfold.utils.tensor_utils import tensor_tree_map
from fastfold.workflow.template import (FastFoldDataWorkFlow, FastFoldMultimerDataWorkFlow)


@contextlib.contextmanager
def temp_fasta_file(fasta_str: str):
    with tempfile.NamedTemporaryFile('w', suffix='.fasta') as fasta_file:
        fasta_file.write(fasta_str)
        fasta_file.seek(0)
        yield fasta_file.name


def add_data_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        '--uniref90_database_path',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--mgnify_database_path',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--pdb70_database_path',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--uniclust30_database_path',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--bfd_database_path',
        type=str,
        default=None,
    )
    parser.add_argument(
        "--pdb_seqres_database_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--uniprot_database_path",
        type=str,
        default=None,
    )
    parser.add_argument('--jackhmmer_binary_path', type=str, default='/usr/bin/jackhmmer')
    parser.add_argument('--hhblits_binary_path', type=str, default='/usr/bin/hhblits')
    parser.add_argument('--hhsearch_binary_path', type=str, default='/usr/bin/hhsearch')
    parser.add_argument('--kalign_binary_path', type=str, default='/usr/bin/kalign')
    parser.add_argument("--hmmsearch_binary_path", type=str, default="hmmsearch")
    parser.add_argument("--hmmbuild_binary_path", type=str, default="hmmbuild")
    parser.add_argument(
        '--max_template_date',
        type=str,
        default=date.today().strftime("%Y-%m-%d"),
    )
    parser.add_argument('--obsolete_pdbs_path', type=str, default=None)
    parser.add_argument('--release_dates_path', type=str, default=None)
    parser.add_argument('--chunk_size', type=int, default=None)
    parser.add_argument('--enable_workflow',
                        default=False,
                        action='store_true',
                        help='run inference with ray workflow or not')
    parser.add_argument('--inplace', default=False, action='store_true')


def inference_model(rank, world_size, result_q, batch, args):
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    # init distributed for Dynamic Axial Parallelism
    habana.enable_habana()
    init_dist()

    device = torch.device("hpu")

    config = model_config(args.model_name)
    if args.chunk_size:
        config.globals.chunk_size = args.chunk_size

    if "v3" in args.param_path:
        set_fused_triangle_multiplication()

    config.globals.inplace = False
    config.globals.is_multimer = args.model_preset == 'multimer'
    model = AlphaFold(config)
    import_jax_weights_(model, args.param_path, version=args.model_name)

    model = inject_habana(model)
    model = model.eval()
    model = model.to(device=device)

    set_chunk_size(model.globals.chunk_size)

    with torch.no_grad():
        batch = {k: torch.as_tensor(v).to(device=device) for k, v in batch.items()}

        t = time.perf_counter()
        out = model(batch)
        htcore.mark_step()
        print(f"Inference time: {time.perf_counter() - t}")

    out = tensor_tree_map(lambda x: np.array(x.cpu()), out)

    result_q.put(out)

    torch.distributed.barrier()


def main(args):
    if args.model_preset == "multimer":
        inference_multimer_model(args)
    else:
        inference_monomer_model(args)


def inference_multimer_model(args):
    print("running in multimer mode...")
    config = model_config(args.model_name)

    predict_max_templates = 4

    template_featurizer = templates.HmmsearchHitFeaturizer(
        mmcif_dir=args.template_mmcif_dir,
        max_template_date=args.max_template_date,
        max_hits=predict_max_templates,
        kalign_binary_path=args.kalign_binary_path,
        release_dates_path=args.release_dates_path,
        obsolete_pdbs_path=args.obsolete_pdbs_path,
    )

    if (not args.use_precomputed_alignments):
        if args.enable_workflow:
            print("Running alignment with ray workflow...")
            alignment_runner = FastFoldMultimerDataWorkFlow(
                jackhmmer_binary_path=args.jackhmmer_binary_path,
                hhblits_binary_path=args.hhblits_binary_path,
                hmmsearch_binary_path=args.hmmsearch_binary_path,
                hmmbuild_binary_path=args.hmmbuild_binary_path,
                uniref90_database_path=args.uniref90_database_path,
                mgnify_database_path=args.mgnify_database_path,
                bfd_database_path=args.bfd_database_path,
                uniclust30_database_path=args.uniclust30_database_path,
                uniprot_database_path=args.uniprot_database_path,
                pdb_seqres_database_path=args.pdb_seqres_database_path,
                use_small_bfd=(args.bfd_database_path is None),
                no_cpus=args.cpus)
        else:
            alignment_runner = data_pipeline.AlignmentRunnerMultimer(
                jackhmmer_binary_path=args.jackhmmer_binary_path,
                hhblits_binary_path=args.hhblits_binary_path,
                hmmsearch_binary_path=args.hmmsearch_binary_path,
                hmmbuild_binary_path=args.hmmbuild_binary_path,
                uniref90_database_path=args.uniref90_database_path,
                mgnify_database_path=args.mgnify_database_path,
                bfd_database_path=args.bfd_database_path,
                uniclust30_database_path=args.uniclust30_database_path,
                uniprot_database_path=args.uniprot_database_path,
                pdb_seqres_database_path=args.pdb_seqres_database_path,
                use_small_bfd=(args.bfd_database_path is None),
                no_cpus=args.cpus)
    else:
        alignment_runner = None

    monomer_data_processor = data_pipeline.DataPipeline(template_featurizer=template_featurizer,)

    data_processor = data_pipeline.DataPipelineMultimer(
        monomer_data_pipeline=monomer_data_processor,)

    output_dir_base = args.output_dir
    random_seed = args.data_random_seed
    if random_seed is None:
        random_seed = random.randrange(sys.maxsize)

    feature_processor = feature_pipeline.FeaturePipeline(config.data)

    if not os.path.exists(output_dir_base):
        os.makedirs(output_dir_base)
    if (not args.use_precomputed_alignments):
        alignment_dir = os.path.join(output_dir_base, "alignments")
    else:
        alignment_dir = args.use_precomputed_alignments

    # Gather input sequences
    fasta_path = args.fasta_path
    with open(fasta_path, "r") as fp:
        data = fp.read()

    lines = [l.replace('\n', '') for prot in data.split('>') for l in prot.strip().split('\n', 1)
            ][1:]
    tags, seqs = lines[::2], lines[1::2]

    for tag, seq in zip(tags, seqs):
        local_alignment_dir = os.path.join(alignment_dir, tag)
        if (args.use_precomputed_alignments is None):
            if not os.path.exists(local_alignment_dir):
                os.makedirs(local_alignment_dir)
            else:
                shutil.rmtree(local_alignment_dir)
                os.makedirs(local_alignment_dir)

            chain_fasta_str = f'>chain_{tag}\n{seq}\n'
            with temp_fasta_file(chain_fasta_str) as chain_fasta_path:
                if args.enable_workflow:
                    print("Running alignment with ray workflow...")
                    t = time.perf_counter()
                    alignment_runner.run(chain_fasta_path, alignment_dir=local_alignment_dir)
                    print(f"Alignment data workflow time: {time.perf_counter() - t}")
                else:
                    alignment_runner.run(chain_fasta_path, local_alignment_dir)

                print(f"Finished running alignment for {tag}")

    local_alignment_dir = alignment_dir

    feature_dict = data_processor.process_fasta(fasta_path=fasta_path,
                                                alignment_dir=local_alignment_dir)
    # feature_dict = pickle.load(open("/home/lcmql/data/features_pdb1o5d.pkl", "rb"))

    processed_feature_dict = feature_processor.process_features(
        feature_dict,
        mode='predict',
        is_multimer=True,
    )

    batch = processed_feature_dict

    manager = mp.Manager()
    result_q = manager.Queue()
    torch.multiprocessing.spawn(inference_model,
                                nprocs=args.hpus,
                                args=(args.hpus, result_q, batch, args))

    out = result_q.get()

    # Toss out the recycling dimensions --- we don't need them anymore
    batch = tensor_tree_map(lambda x: np.array(x[..., -1].cpu()), batch)

    plddt = out["plddt"]
    mean_plddt = np.mean(plddt)

    plddt_b_factors = np.repeat(plddt[..., None], residue_constants.atom_type_num, axis=-1)

    unrelaxed_protein = protein.from_prediction(features=batch,
                                                result=out,
                                                b_factors=plddt_b_factors)

    # Save the unrelaxed PDB.
    unrelaxed_output_path = os.path.join(args.output_dir, f'{tag}_{args.model_name}_unrelaxed.pdb')
    with open(unrelaxed_output_path, 'w') as f:
        f.write(protein.to_pdb(unrelaxed_protein))

    amber_relaxer = relax.AmberRelaxation(
        use_gpu=False,
        **config.relax,
    )

    # Relax the prediction.
    t = time.perf_counter()
    relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)
    print(f"Relaxation time: {time.perf_counter() - t}")

    # Save the relaxed PDB.
    relaxed_output_path = os.path.join(args.output_dir, f'{tag}_{args.model_name}_relaxed.pdb')
    with open(relaxed_output_path, 'w') as f:
        f.write(relaxed_pdb_str)


def inference_monomer_model(args):
    print("running in monomer mode...")
    config = model_config(args.model_name)

    template_featurizer = templates.TemplateHitFeaturizer(
        mmcif_dir=args.template_mmcif_dir,
        max_template_date=args.max_template_date,
        max_hits=config.data.predict.max_templates,
        kalign_binary_path=args.kalign_binary_path,
        release_dates_path=args.release_dates_path,
        obsolete_pdbs_path=args.obsolete_pdbs_path)

    use_small_bfd = args.preset == 'reduced_dbs'  # (args.bfd_database_path is None)
    if use_small_bfd:
        assert args.bfd_database_path is not None
    else:
        assert args.bfd_database_path is not None
        assert args.uniclust30_database_path is not None

    data_processor = data_pipeline.DataPipeline(template_featurizer=template_featurizer,)

    output_dir_base = args.output_dir
    random_seed = args.data_random_seed
    if random_seed is None:
        random_seed = random.randrange(sys.maxsize)
    feature_processor = feature_pipeline.FeaturePipeline(config.data)
    if not os.path.exists(output_dir_base):
        os.makedirs(output_dir_base)
    if (args.use_precomputed_alignments is None):
        alignment_dir = os.path.join(output_dir_base, "alignments")
    else:
        alignment_dir = args.use_precomputed_alignments

    # Gather input sequences
    with open(args.fasta_path, "r") as fp:
        fasta = fp.read()
    seqs, tags = parse_fasta(fasta)
    seq, tag = seqs[0], tags[0]

    print(f"tag:{tag}\nseq[{len(seq)}]:{seq}")
    batch = [None]

    fasta_path = os.path.join(args.output_dir, "tmp.fasta")
    with open(fasta_path, "w") as fp:
        fp.write(f">{tag}\n{seq}")

    print("Generating features...")
    local_alignment_dir = os.path.join(alignment_dir, tag)

    if (args.use_precomputed_alignments is None):
        if not os.path.exists(local_alignment_dir):
            os.makedirs(local_alignment_dir)
        if args.enable_workflow:
            print("Running alignment with ray workflow...")
            alignment_data_workflow_runner = FastFoldDataWorkFlow(
                jackhmmer_binary_path=args.jackhmmer_binary_path,
                hhblits_binary_path=args.hhblits_binary_path,
                hhsearch_binary_path=args.hhsearch_binary_path,
                uniref90_database_path=args.uniref90_database_path,
                mgnify_database_path=args.mgnify_database_path,
                bfd_database_path=args.bfd_database_path,
                uniclust30_database_path=args.uniclust30_database_path,
                pdb70_database_path=args.pdb70_database_path,
                use_small_bfd=use_small_bfd,
                no_cpus=args.cpus,
            )
            t = time.perf_counter()
            alignment_data_workflow_runner.run(fasta_path, alignment_dir=local_alignment_dir)
            print(f"Alignment data workflow time: {time.perf_counter() - t}")
        else:
            alignment_runner = data_pipeline.AlignmentRunner(
                jackhmmer_binary_path=args.jackhmmer_binary_path,
                hhblits_binary_path=args.hhblits_binary_path,
                hhsearch_binary_path=args.hhsearch_binary_path,
                uniref90_database_path=args.uniref90_database_path,
                mgnify_database_path=args.mgnify_database_path,
                bfd_database_path=args.bfd_database_path,
                uniclust30_database_path=args.uniclust30_database_path,
                pdb70_database_path=args.pdb70_database_path,
                use_small_bfd=use_small_bfd,
                no_cpus=args.cpus,
            )
            alignment_runner.run(fasta_path, local_alignment_dir)

    feature_dict = data_processor.process_fasta(fasta_path=fasta_path,
                                                alignment_dir=local_alignment_dir)

    # Remove temporary FASTA file
    os.remove(fasta_path)

    processed_feature_dict = feature_processor.process_features(
        feature_dict,
        mode='predict',
    )

    batch = processed_feature_dict

    manager = mp.Manager()
    result_q = manager.Queue()
    torch.multiprocessing.spawn(inference_model,
                                nprocs=args.hpus,
                                args=(args.hpus, result_q, batch, args))

    out = result_q.get()

    # Toss out the recycling dimensions --- we don't need them anymore
    batch = tensor_tree_map(lambda x: np.array(x[..., -1].cpu()), batch)

    plddt = out["plddt"]
    mean_plddt = np.mean(plddt)

    plddt_b_factors = np.repeat(plddt[..., None], residue_constants.atom_type_num, axis=-1)

    unrelaxed_protein = protein.from_prediction(features=batch,
                                                result=out,
                                                b_factors=plddt_b_factors)

    # Save the unrelaxed PDB.
    unrelaxed_output_path = os.path.join(args.output_dir, f'{tag}_{args.model_name}_unrelaxed.pdb')
    with open(unrelaxed_output_path, 'w') as f:
        f.write(protein.to_pdb(unrelaxed_protein))

    amber_relaxer = relax.AmberRelaxation(
        use_gpu=False,
        **config.relax,
    )

    # Relax the prediction.
    t = time.perf_counter()
    relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)
    print(f"Relaxation time: {time.perf_counter() - t}")

    # Save the relaxed PDB.
    relaxed_output_path = os.path.join(args.output_dir, f'{tag}_{args.model_name}_relaxed.pdb')
    with open(relaxed_output_path, 'w') as f:
        f.write(relaxed_pdb_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fasta_path",
        type=str,
    )
    parser.add_argument(
        "template_mmcif_dir",
        type=str,
    )
    parser.add_argument("--use_precomputed_alignments",
                        type=str,
                        default=None,
                        help="""Path to alignment directory. If provided, alignment computation 
                is skipped and database path arguments are ignored.""")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.getcwd(),
        help="""Name of the directory in which to output the prediction""",
    )
    parser.add_argument("--model_name",
                        type=str,
                        default="model_1",
                        help="""Name of a model config. Choose one of model_{1-5} or 
             model_{1-5}_ptm or model_{1-5}_multimer, as defined on the AlphaFold GitHub.""")
    parser.add_argument("--param_path",
                        type=str,
                        default=None,
                        help="""Path to model parameters. If None, parameters are selected
             automatically according to the model name from 
             ./data/params""")
    parser.add_argument("--cpus",
                        type=int,
                        default=12,
                        help="""Number of CPUs with which to run alignment tools""")
    parser.add_argument("--hpus",
                        type=int,
                        default=1,
                        help="""Number of GPUs with which to run inference""")
    parser.add_argument('--preset',
                        type=str,
                        default='full_dbs',
                        choices=('reduced_dbs', 'full_dbs'))
    parser.add_argument('--data_random_seed', type=str, default=None)
    parser.add_argument(
        "--model_preset",
        type=str,
        default="monomer",
        choices=["monomer", "multimer"],
        help="Choose preset model configuration - the monomer model, the monomer model with "
        "extra ensembling, monomer model with pTM head, or multimer model",
    )
    add_data_args(parser)
    args = parser.parse_args()

    if (args.param_path is None):
        args.param_path = os.path.join("data", "params", "params_" + args.model_name + ".npz")

    main(args)
