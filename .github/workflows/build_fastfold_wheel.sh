#!/usr/bin/env bash

method=${1}
url=${2}
filename=${3}
cuda_version=${4}
python_version=${5}
torch_version=${6}
flags=${@:7}

git reset --hard HEAD
source activate base
conda create -n $python_version -y python=$python_version
source activate $python_version

if [ $1 == "pip" ]
then
    wget -nc -q -O ./$filename $url
    pip install ./$filename
    
elif [ $1 == 'conda' ]
then
    conda install pytorch==$torch_version cudatoolkit=$cuda_version $flags
else
    echo Invalid installation method
    exit
fi

python setup.py bdist_wheel
python setup.py clean
conda env remove -n $python_version