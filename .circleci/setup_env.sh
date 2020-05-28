#!/bin/bash
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
bash miniconda.sh -b -p ~/miniconda
export PATH=~/miniconda/bin:$PATH
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda install -q conda-build
conda create -n bld --override-channels -c intel daal daal-devel
conda install -q -n bld --override-channels -c defaults python=3.7 numpy scipy pytest pandas pyyaml joblib numpydoc
conda install -q -n bld -c conda-forge --override-channels mpi mpich
gcc -v
g++ -v
head /proc/cpuinfo
