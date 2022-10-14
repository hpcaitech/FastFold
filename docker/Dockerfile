FROM hpcaitech/colossalai:0.1.8

RUN conda install openmm=7.7.0 pdbfixer -c conda-forge -y \
 && conda install hmmer==3.3.2 hhsuite=3.3.0 kalign2=2.04 -c bioconda -y

RUN pip install biopython==1.79 dm-tree==0.1.6 ml-collections==0.1.0 \
scipy==1.7.1 ray pyarrow pandas einops

# prepare environment
Run git clone https://github.com/hpcaitech/FastFold.git\
 && cd ./FastFold \
 && python setup.py install
