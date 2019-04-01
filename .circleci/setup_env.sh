#!/bin/bash
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
bash miniconda.sh -b -p ~/miniconda
export PATH=~/miniconda/bin:$PATH
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda install -q conda-build
conda create -q -n bld --override-channels -c intel -c conda-forge python=3.6 numpy scipy pytest daal tbb pandas
conda install -q -n bld -c defaults --override-channels --no-deps libgcc-ng libstdcxx-ng
conda install -q -n bld -c conda-forge --override-channels --no-deps mpi libgfortran mpich
conda install -c defaults pyyaml
