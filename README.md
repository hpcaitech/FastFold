![](/assets/fold.jpg)

# FastFold

[![](https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green)](https://arxiv.org/abs/2203.00854)
![](https://img.shields.io/badge/Made%20with-ColossalAI-blueviolet?style=flat)
![](https://img.shields.io/github/v/release/hpcaitech/FastFold)
[![GitHub license](https://img.shields.io/github/license/hpcaitech/FastFold)](https://github.com/hpcaitech/FastFold/blob/main/LICENSE)

Optimizing Protein Structure Prediction Model Training and Inference on GPU Clusters

FastFold provides a **high-performance implementation of Evoformer** with the following characteristics.

1. Excellent kernel performance on GPU platform
2. Supporting Dynamic Axial Parallelism(DAP)
    * Break the memory limit of single GPU and reduce the overall training time
    * DAP can significantly speed up inference and make ultra-long sequence inference possible
3. Ease of use
    * Huge performance gains with a few lines changes
    * You don't need to care about how the parallel part is implemented

## Installation

You will need Python 3.8 or later and [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads) 11.1 or above when you are installing from source. 

We highly recommend installing an Anaconda or Miniconda environment and install PyTorch with conda:

```
conda create -n fastfold python=3.8
conda activate fastfold
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

You can get the FastFold source and install it with setuptools:

```shell
git clone https://github.com/hpcaitech/FastFold
cd FastFold
python setup.py install
```

## Usage

You can use `Evoformer` as `nn.Module` in your project after `from fastfold.model import Evoformer`:

```python
from fastfold.model import Evoformer
evoformer_layer = Evoformer()
```

If you want to use Dynamic Axial Parallelism, add a line of initialize with `fastfold.distributed.init_dap`.

```python
from fastfold.distributed import init_dap

init_dap(args.dap_size)
```

### Inference

You can use FastFold alongwith OpenFold with `inject_openfold`. This will replace the evoformer in OpenFold with the high performance evoformer from FastFold.

```python
from fastfold.utils import inject_openfold

model = AlphaFold(config)
import_jax_weights_(model, args.param_path, version=args.model_name)

model = inject_openfold(model)
```

For Dynamic Axial Parallelism, you can refer to `./inference.py`. Here is an example of 2 GPUs parallel inference:

```shell
torchrun --nproc_per_node=2 inference.py target.fasta data/pdb_mmcif/mmcif_files/ \
    --uniref90_database_path data/uniref90/uniref90.fasta \
    --mgnify_database_path data/mgnify/mgy_clusters_2018_12.fa \
    --pdb70_database_path data/pdb70/pdb70 \
    --uniclust30_database_path data/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
    --output_dir ./ \
    --bfd_database_path data/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
    --jackhmmer_binary_path lib/conda/envs/openfold_venv/bin/jackhmmer \
    --hhblits_binary_path lib/conda/envs/openfold_venv/bin/hhblits \
    --hhsearch_binary_path lib/conda/envs/openfold_venv/bin/hhsearch \
    --kalign_binary_path lib/conda/envs/openfold_venv/bin/kalign
```

## Performance Benchmark

We have included a performance benchmark script in `./benchmark`. You can benchmark the performance of Evoformer using different settings.

```shell
cd ./benchmark
torchrun --nproc_per_node=1 perf.py --msa-length 128 --res-length 256
```

Benchmark Dynamic Axial Parallelism with 2 GPUs:

```shell
cd ./benchmark
torchrun --nproc_per_node=2 perf.py --msa-length 128 --res-length 256 --dap-size 2
```

If you want to benchmark with [OpenFold](https://github.com/aqlaboratory/openfold), you need to install OpenFold first and benchmark with option `--openfold`:

```shell
torchrun --nproc_per_node=1 perf.py --msa-length 128 --res-length 256 --openfold
```

## Cite us

Cite this paper, if you use FastFold in your research publication.

```
@misc{cheng2022fastfold,
      title={FastFold: Reducing AlphaFold Training Time from 11 Days to 67 Hours}, 
      author={Shenggan Cheng and Ruidong Wu and Zhongming Yu and Binrui Li and Xiwen Zhang and Jian Peng and Yang You},
      year={2022},
      eprint={2203.00854},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
