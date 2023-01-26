#!/bin/bash

python -u ./evaluate.py \
	--cameo_path ./cameo/data_dir/ \
        --alignment_dir ./cameo/alignment_dir/ \
	--ckpt_path ./model.20.pth \
        --template_mmcif_dir ./data/pdb_mmcif/mmcif_files \
        --kalign_binary_path `which kalign`

