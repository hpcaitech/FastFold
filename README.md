![](/assets/fold.jpg)

# FastFold

[![](https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green)](https://arxiv.org/abs/2203.00854)
![](https://img.shields.io/badge/Made%20with-ColossalAI-blueviolet?style=flat)
![](https://img.shields.io/badge/Habana-support-blue?style=flat&logo=intel&logoColor=blue)
![](https://img.shields.io/github/v/release/hpcaitech/FastFold)
[![GitHub license](https://img.shields.io/github/license/hpcaitech/FastFold)](https://github.com/hpcaitech/FastFold/blob/main/LICENSE)

## News :triangular_flag_on_post:
- [2023/01] Compatible with AlphaFold v2.3
- [2023/01] Added support for inference and training of AlphaFold on [Intel Habana](https://habana.ai/) platform. For usage instructions, see [here](#Inference-or-Training-on-Intel-Habana).

<br>

Optimizing Protein Structure Prediction Model Training and Inference on Heterogeneous Clusters

FastFold provides a **high-performance implementation of Evoformer** with the following characteristics.

1. Excellent kernel performance on GPU platform
2. Supporting Dynamic Axial Parallelism(DAP)
    * Break the memory limit of single GPU and reduce the overall training time
    * DAP can significantly speed up inference and make ultra-long sequence inference possible
3. Ease of use
    * Huge performance gains with a few lines changes
    * You don't need to care about how the parallel part is implemented
4. Faster data processing, about 3x times faster on monomer, about 3Nx times faster on multimer with N sequence.
5. Great Reduction on GPU memory, able to inference sequence containing more than **10000** residues.

## Installation

To install FastFold, you will need:
+ Python 3.8 or 3.9.
+ [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads) 11.3 or above
+ PyTorch 1.12 or above 

For now, You can install FastFold:
### Using Conda (Recommended)

We highly recommend installing an Anaconda or Miniconda environment and install PyTorch with conda.
Lines below would create a new conda environment called "fastfold":

```shell
git clone https://github.com/hpcaitech/FastFold
cd FastFold
conda env create --name=fastfold -f environment.yml
conda activate fastfold
python setup.py install
```

#### Advanced

To leverage the power of FastFold, we recommend you to install [Triton](https://github.com/openai/triton).

**NOTE: Triron needs CUDA 11.4 to run.**

```bash
pip install -U --pre triton
```


## Use Docker

### Build On Your Own
Run the following command to build a docker image from Dockerfile provided.

> Building FastFold from scratch requires GPU support, you need to use Nvidia Docker Runtime as the default when doing `docker build`. More details can be found [here](https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime).

```shell
cd FastFold
docker build -t fastfold ./docker
```

Run the following command to start the docker container in interactive mode.
```shell
docker run -ti --gpus all --rm --ipc=host fastfold bash
```

## Usage

You can use `Evoformer` as `nn.Module` in your project after `from fastfold.model.fastnn import Evoformer`:

```python
from fastfold.model.fastnn import Evoformer
evoformer_layer = Evoformer()
```

If you want to use Dynamic Axial Parallelism, add a line of initialize with `fastfold.distributed.init_dap`.

```python
from fastfold.distributed import init_dap

init_dap(args.dap_size)
```

### Download the dataset
You can down the dataset used to train FastFold  by the script `download_all_data.sh`:

    ./scripts/download_all_data.sh data/

### Inference

You can use FastFold with `inject_fastnn`. This will replace the evoformer from OpenFold with the high performance evoformer from FastFold.

```python
from fastfold.utils import inject_fastnn

model = AlphaFold(config)
import_jax_weights_(model, args.param_path, version=args.model_name)

model = inject_fastnn(model)
```

For Dynamic Axial Parallelism, you can refer to `./inference.py`. Here is an example of 2 GPUs parallel inference:

```shell
python inference.py target.fasta data/pdb_mmcif/mmcif_files/ \
    --output_dir .outputs/ \
    --gpus 2 \
    --uniref90_database_path data/uniref90/uniref90.fasta \
    --mgnify_database_path data/mgnify/mgy_clusters_2022_05.fa \
    --pdb70_database_path data/pdb70/pdb70 \
    --uniref30_database_path data/uniref30/UniRef30_2021_03 \
    --bfd_database_path data/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
    --jackhmmer_binary_path `which jackhmmer` \
    --hhblits_binary_path `which hhblits` \
    --hhsearch_binary_path `which hhsearch` \
    --kalign_binary_path `which kalign` \
    --enable_workflow \
    --inplace
```
or run the script `./inference.sh`, you can change the parameter in the script, especisally those data path.
```shell
./inference.sh
```

Alphafold's data pre-processing takes a lot of time, so we speed up the data pre-process by [ray](https://docs.ray.io/en/latest/workflows/concepts.html) workflow, which achieves a 3x times faster speed. To run the inference with ray workflow, we add parameter `--enable_workflow` by default.
To reduce memory usage of embedding presentations, we also add parameter `--inplace` to share memory by defaul.

#### inference with lower memory usage
Alphafold's embedding presentations take up a lot of memory as the sequence length increases. To reduce memory usage, 
you should add parameter `--chunk_size [N]` to cmdline or shell script `./inference.sh`. 
The smaller you set N, the less memory will be used, but it will affect the speed. We can inference 
a sequence of length 10000 in bf16 with 61GB memory on a Nvidia A100(80GB). For fp32, the max length is 8000.
> You need to set `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:15000` to inference such an extreme long sequence.

```shell
python inference.py target.fasta data/pdb_mmcif/mmcif_files/ \
    --output_dir .outputs/ \
    --gpus 2 \
    --uniref90_database_path data/uniref90/uniref90.fasta \
    --mgnify_database_path data/mgnify/mgy_clusters_2022_05.fa \
    --pdb70_database_path data/pdb70/pdb70 \
    --uniref30_database_path data/uniref30/UniRef30_2021_03 \
    --bfd_database_path data/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
    --jackhmmer_binary_path `which jackhmmer` \
    --hhblits_binary_path `which hhblits` \
    --hhsearch_binary_path `which hhsearch` \
    --kalign_binary_path `which kalign`  \
    --enable_workflow \
    --inplace
    --chunk_size N \
```

#### inference multimer sequence
Alphafold Multimer is supported. You can the following cmd or shell script `./inference_multimer.sh`.
Workflow and memory parameters mentioned above can also be used.
```shell
python inference.py target.fasta data/pdb_mmcif/mmcif_files/ \
    --output_dir ./ \
    --gpus 2 \
    --model_preset multimer \
    --uniref90_database_path data/uniref90/uniref90.fasta \
    --mgnify_database_path data/mgnify/mgy_clusters_2022_05.fa \
    --pdb70_database_path data/pdb70/pdb70 \
    --uniref30_database_path data/uniref30/UniRef30_2021_03 \
    --bfd_database_path data/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
    --uniprot_database_path data/uniprot/uniprot.fasta \
    --pdb_seqres_database_path data/pdb_seqres/pdb_seqres.txt  \
    --param_path data/params/params_model_1_multimer.npz \
    --model_name model_1_multimer \
    --jackhmmer_binary_path `which jackhmmer` \
    --hhblits_binary_path `which hhblits` \
    --hhsearch_binary_path `which hhsearch` \
    --kalign_binary_path `which kalign`
```

### Inference or Training on Intel Habana

To run AlphaFold inference or training on Intel Habana, you can follow the instructions in the [Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/) to set up your environment on Amazon EC2 DL1 instances or on-premise environments, and please use SynapseAI R1.7.1 to test as it was verified internally.

Once you have prepared your dataset and installed fastfold, you can use the following scripts:

```shell
cd fastfold/habana/fastnn/custom_op/; python setup.py build (this is for Gaudi, for Gaudi2 please use setup2.py) ; cd -
bash habana/inference.sh
bash habana/train.sh
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

## Acknowledgments

We would like to extend our special thanks to the Intel Habana team for their support in providing us with technology and resources on the Habana platform.
