# add '--gpus [N]' to use N gpus for inference
# add '--enable_workflow' to use parallel workflow for data processing
# add '--use_precomputed_alignments [path_to_alignments]' to use precomputed msa
# add '--chunk_size [N]' to use chunk to reduce peak memory
# add '--inplace' to use inplace to save memory

python inference.py target.fasta data/pdb_mmcif/mmcif_files \
    --output_dir ./ \
    --gpus 1 \
    --uniref90_database_path data/uniref90/uniref90.fasta \
    --mgnify_database_path data/mgnify/mgy_clusters_2022_05.fa \
    --pdb70_database_path data/pdb70/pdb70 \
    --pdb_seqres_database_path data/pdb_seqres/pdb_seqres.txt \
    --uniprot_database_path data/uniprot/uniprot.fasta \
    --uniref30_database_path data/uniref30/UniRef30_2021_03 \
    --bfd_database_path data/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
    --jackhmmer_binary_path `which jackhmmer` \
    --hhblits_binary_path `which hhblits` \
    --hhsearch_binary_path `which hhsearch` \
    --kalign_binary_path `which kalign`  \
    --model_preset multimer \
    --param_path data/params/params_model_1_multimer_v3.npz \
    --model_name model_1_multimer \
