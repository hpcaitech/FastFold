FROM vault.habana.ai/gaudi-docker/1.14.0/ubuntu22.04/habanalabs/pytorch-installer-2.1.1:latest

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh -b
RUN source ${HOME}/miniconda3/etc/profile.d/conda.sh && conda create -n habana python=3.10.12 && conda activate habana && conda install openmm=7.7.0 pdbfixer -c conda-forge -y \
 && conda install hmmer==3.3.2 hhsuite=3.3.0 kalign2=2.04 -c bioconda -y

RUN source ${HOME}/miniconda3/etc/profile.d/conda.sh && conda activate habana && python -m pip install biopython==1.79 dm-tree==0.1.6 ml-collections==0.1.0 \
scipy==1.7.1 ray pyarrow pandas einops colossalai

ENV PYTHONPTAH=$PYTHONPATH:/usr/local/lib/python3.10/dist-packages

Run source ${HOME}/miniconda3/etc/profile.d/conda.sh && conda activate habana && git clone https://github.com/hpcaitech/FastFold.git \
 && cd ./FastFold \
 && python setup.py install \
 && cd fastfold/habana/fastnn/custom_op/ && python ./setup2.py build && cd -
