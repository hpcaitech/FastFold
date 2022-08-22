rm -rf alignments/
rm -rf *.pdb
python inference_with_workflow.py target.fasta /data/scratch/alphafold/alphafold/pdb_mmcif/mmcif_files \
    --output_dir ./ \
    --gpus 2 \
    --uniref90_database_path /data/scratch/alphafold/alphafold/uniref90/uniref90.fasta \
    --mgnify_database_path /data/scratch/alphafold/alphafold/mgnify/mgy_clusters_2018_12.fa \
    --pdb70_database_path /data/scratch/alphafold/alphafold/pdb70/pdb70 \
    --param_path /data/scratch/alphafold/alphafold/params/params_model_1.npz \
    --uniclust30_database_path /data/scratch/alphafold/alphafold/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
    --bfd_database_path /data/scratch/alphafold/alphafold/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
    --jackhmmer_binary_path `which jackhmmer` \
    --hhblits_binary_path `which hhblits` \
    --hhsearch_binary_path `which hhsearch` \
    --kalign_binary_path `which kalign` 