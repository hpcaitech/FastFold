DATA_DIR=/path/to/data
PROJECT_DIR=/path/to/project

gpus_per_node=2
nnodes=1

max_template_date=2021-10-10

train_data_dir=${DATA_DIR}/mmcif_dir  # specify the dir contains *.cif or *.pdb
train_alignment_dir=${DATA_DIR}/alignment_dir  # a dir to save template and features.pkl of training sequence
mkdir -p ${train_alignment_dir}

# val_data_dir=${PROJECT_DIR}/dataset/val_pdb
# val_alignment_dir=${PROJECT_DIR}/dataset/alignment_val_pdb  # a dir to save template and features.pkl of vld sequence

template_mmcif_dir=${DATA_DIR}/data/pdb_mmcif/mmcif_files
template_release_dates_cache_path=${DATA_DIR}/mmcif_cache.json  # a cache used to pre-filter templates
train_chain_data_cache_path=${DATA_DIR}/chain_data_cache.json  # a separate chain-level cache with data used for training-time data filtering

train_epoch_len=10000  # virtual length of each training epoch, which affects frequency of validation & checkpointing

torchrun --standalone --nproc_per_node ${gpus_per_node} --nnodes ${nnodes} train.py \
    --from_torch \
    --template_mmcif_dir=${template_mmcif_dir} \
    --max_template_date=${max_template_date} \
    --train_data_dir=${train_data_dir} \
    --train_alignment_dir=${train_alignment_dir} \
    --train_chain_data_cache_path=${train_chain_data_cache_path} \
    --template_release_dates_cache_path=${template_release_dates_cache_path} \
    --train_epoch_len=${train_epoch_len} \
