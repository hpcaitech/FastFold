#!/usr/bin/env bash

url=${1}
filename=${2}
cuda_version=${3}
python_version=${4}

git reset --hard HEAD
mkdir -p ./all_dist
source activate base
conda create -n $python_version -y python=$python_version
source activate $python_version

wget -nc -q -O ./$filename $url
pip install ./$filename
pip install numpy

python setup.py bdist_wheel
mv ./dist/* ./all_dist
python setup.py clean
conda deactivate
conda env remove -n $python_version