#!/bin/bash
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
bash miniconda.sh -b -p ~/miniconda
export PATH=~/miniconda/bin:$PATH
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda create -n bld python=3.7 conda-build
source activate bld
conda install -q scipy pytest pandas pyyaml joblib numpydoc cython jinja2 numpy clang-tools
conda install -q --override-channels -c intel dal dal-include
gcc -v
g++ -v
head /proc/cpuinfo
