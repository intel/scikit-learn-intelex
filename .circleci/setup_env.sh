#!/bin/bash
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
bash miniconda.sh -b -p ~/miniconda
export PATH=~/miniconda/bin:$PATH
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda install -q conda-build
conda create -n bld --override-channels -c intel daal daal-devel tbb
conda install -q -n bld --override-channels -c defaults python=3.6 numpy scipy pytest pandas pyyaml joblib
conda install -q -n bld -c defaults --override-channels --no-deps libgcc-ng libstdcxx-ng
conda install -q -n bld -c conda-forge --override-channels numpydocs
conda install -q -n bld -c conda-forge --override-channels --no-deps mpi libgfortran mpich
gcc -v
g++ -v
head /proc/cpuinfo
