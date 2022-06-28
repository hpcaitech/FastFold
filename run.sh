#!/bin/bash

torchrun --nproc_per_node=2 inference.py \
    test/input.fasta \
    /data/scratch/alphafold/alphafold/pdb_mmcif/mmcif_files/ \
    --output_dir ./output/ \
    --param_path /data/scratch/alphafold/alphafold/params/params_model_1.npz \
    --uniref90_database_path /data/scratch/alphafold/alphafold/uniref90/uniref90.fasta \
    --mgnify_database_path /data/scratch/alphafold/alphafold/mgnify/mgy_clusters_2018_12.fa \
    --pdb70_database_path /data/scratch/alphafold/alphafold/pdb70/pdb70 \
    --uniclust30_database_path /data/scratch/alphafold/alphafold/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
    --bfd_database_path /data/scratch/alphafold/rosettafold/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
    --jackhmmer_binary_path `which jackhmmer` \
    --hhblits_binary_path `which hhblits` \
    --hhsearch_binary_path `which hhsearch` \
    --kalign_binary_path `which kalign`