FROM hpcaitech/cuda-conda:11.3


# install dependency
RUN yum install -y patch

RUN conda install pytorch==1.10.0 torchvision torchaudio cudatoolkit=11.3 -c pytorch \
 && conda install setuptools=59.5.0 openmm=7.5.1 pdbfixer -c conda-forge \
 && conda install hmmer==3.3.2 hhsuite=3.3.0 kalign2=2.04 -c bioconda

RUN pip install biopython==1.79 dm-tree==0.1.6 ml-collections==0.1.0 numpy==1.21.2 \
 PyYAML==5.4.1 requests==2.26.0 scipy==1.7.1 tqdm==4.62.2 typing-extensions==3.10.0.2 einops ray==1.13.0 pyarrow

RUN pip install colossalai==0.1.8+torch1.10cu11.3 -f https://release.colossalai.org


# prepare environment
Run git clone https://github.com/hpcaitech/FastFold.git\
 && cd ./FastFold \
 && /bin/bash scripts/patch_openmm.sh \
 && python setup.py install
