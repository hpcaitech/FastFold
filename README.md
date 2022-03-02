![](/assets/fold.jpg)

# FastFold

![](https://img.shields.io/github/v/release/hpcaitech/FastFold)
[![GitHub license](https://img.shields.io/github/license/hpcaitech/FastFold.svg)](https://github.com/hpcaitech/FastFold/blob/master/LICENSE)
![](https://img.shields.io/badge/Made%20with-ColossalAI-blueviolet?style=flat)

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

If you want to use Dynamic Axial Parallelism, add a line of initialize with `fastfold.distributed.init_dap` after `torch.distributed.init_process_group`.

```python
from fastfold.distributed import init_dap

torch.distributed.init_process_group(backend='nccl', init_method='env://')
init_dap(args.dap_size)
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
```
