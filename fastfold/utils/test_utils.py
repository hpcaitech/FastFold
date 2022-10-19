import os


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
