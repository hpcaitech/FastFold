import os
import random

import torch
import numpy as np

def get_param_path():
    # develop
    if os.path.exists('/data/scratch/alphafold/alphafold/params/params_model_1.npz'):
        return '/data/scratch/alphafold/alphafold/params/params_model_1.npz'
    # test
    return '/data/scratch/fastfold/weight.npz'


def get_data_path():
    # develop
    if os.path.exists('/home/lczxl/data2/fastfold/example_input/mono_batch.pkl'):
        return '/home/lczxl/data2/fastfold/example_input/mono_batch.pkl'
    # test
    return '/data/scratch/fastfold/mono_batch.pkl'


def get_train_data_path():
    return '/data/scratch/fastfold/std_train_batch.pkl'

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False