FROM image-cloud.luchentech.com/hpcaitech/pytorch-cuda:2.3.0-12.1.0


RUN apt-get update && \
    apt-get install -y gcc-9 g++-9 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 100 \
                        --slave   /usr/bin/g++ g++ /usr/bin/g++-9 && \
    rm -rf /var/lib/apt/lists/*

ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++

RUN conda update -n base -c defaults -y conda openssl ca-certificates && \
    conda config --remove-key channels || true && \
    conda config --add channels conda-forge && \
    conda config --add channels bioconda && \
    conda config --add channels defaults && \
    conda config --set channel_priority strict && \
    conda config --set show_channel_urls yes

RUN conda install -n pytorch -y -c conda-forge -c bioconda \
        openmm=8.2.0 \
        pdbfixer \
        hmmer=3.3.2 \
        hhsuite=3.3.0 \
        scipy==1.11.4 \
        kalign2=2.04 && \
    conda clean -afy

RUN /opt/conda/envs/pytorch/bin/pip install --no-cache-dir \
        biopython==1.79 \
        dm-tree==0.1.6 \
        ml-collections==0.1.0 \
        ray==2.47.1 \
        pyarrow \
        pandas \
        einops \
        colossalai==0.5.0


RUN git clone --depth 1 https://github.com/hpcaitech/FastFold.git && \
    cd FastFold && \
    /opt/conda/envs/pytorch/bin/python setup.py install && 


WORKDIR /workspace