target_fasta=$1

python inference.py $target_fasta data/alphafold/pdb_mmcif/mmcif_files \
    --output_dir ./outputs/$target_fasta \
    --gpus 1 \
    --uniref90_database_path data/alphafold/uniref90/uniref90.fasta \
    --mgnify_database_path data/alphafold/mgnify/mgy_clusters_2018_12.fa \
    --pdb70_database_path data/alphafold/pdb70/pdb70 \
    --uniclust30_database_path data/alphafold/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
    --bfd_database_path data/alphafold/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
    --jackhmmer_binary_path `which jackhmmer` \
    --hhblits_binary_path `which hhblits` \
    --hhsearch_binary_path `which hhsearch` \
    --kalign_binary_path `which kalign`  \
    --use_precomputed_alignments ./outputs/$target_fasta/alignments \
    --chunk_size 1
