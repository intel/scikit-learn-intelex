# daal4py - A Convenient Python API to the Intel® Data Analytics Acceleration Library (Intel® DAAL)
[![Build Status](https://travis-ci.com/IntelPython/daal4py.svg?branch=master)](https://travis-ci.com/IntelPython/daal4py)


A simplified API to Intel® DAAL that allows for fast usage of the framework suited for Data Scientists or Machine Learning users.  Built to help provide an abstraction to Intel® DAAL for either direct usage or integration into one's own framework.  

- [Documentation](https://intelpython.github.io/daal4py/)
- [Source Code](https://github.com/IntelPython/daal4py/tree/master/src)
- [About Intel® DAAL](https://software.intel.com/en-us/intel-daal)

Running full scikit-learn test suite with daal4p's optimization patches

- [![CircleCI](https://circleci.com/gh/IntelPython/daal4py.svg?style=svg)](https://circleci.com/gh/IntelPython/daal4py) when applied to scikit-learn from PyPi
- [![CircleCI](https://circleci.com/gh/IntelPython/daal4py/tree/test-sklearn-master.svg?style=svg)](https://circleci.com/gh/IntelPython/daal4py/tree/test-sklearn-master) when applied to built master branch

With this daal4py API, your Python programs can use Intel® DAAL algorithms in just one line:
```
kmeans_init(data, 10, t_method="plusPlusDense")
```
You can even run this on a cluster by simple adding a keyword-parameter
```
kmeans_init(data, 10, t_method="plusPlusDense", distributed=True)
```
# Getting Started
daal4py is easily built from source with the majority of the necessary prerequisites available on conda.  The instructions below detail how to gather the prerequisites, setting one's build environment, and finally building and installing the completed package.  daal4py can be built for all three major platforms (Windows, Linux, macOS). Multi-node (distributed) and streaming support can be disabled if desired.  

The build-process (using setup.py) happens in 3 stages:
1. Creating C++ and cython sources from DAAL C++ headers
2. Running cython on generated source
3. Compiling and linking

# Building daal4py using conda-build
The easiest way to build daal4py is using the conda-build withe the provided recipe.

## Prerequisites
* Python version 2.7 or >= 3.6
* conda-build version >= 3
* C++ compiler with C++11 support

## Building daal4py
```
cd <checkout-dir>
conda build conda-recipe -c intel -c conda-forge
```
This will build the conda package and tell you where to find it (```.../daal4py*.tar.bz2```).

## Installing the built daal4py conda package
```
conda install <path-to-conda-package-as-built-above>
```
To actually use your daal4py, dependent packages need to be installed. To ensure, do

Linux and OsX:
```
conda install -c intel -c conda-forge mpich tbb daal numpy
```
Windows:
```
conda install -c intel mpi_rt tbb daal numpy
```

# Building daal4py without conda-build
Without conda-build you have to manually setup your environment before building daal4py.

## Prerequisites
* Python version 2.7 or >= 3.6
* Jinja2
* Cython
* Numpy
* A C++ compiler with C++11 support
* Intel(R) Threading Building Blocks (Intel® TBB) version 2018.0.4 or later (https://www.threadingbuildingblocks.org/)
  * You can use the pre-built conda package from Intel's channel or conda-forge channel on anaconda.org (see below)
  * Needed for distributed mode. You can disable support for distributed mode by setting NO_DIST to '1' or 'yes'
* Intel® Data Analytics Acceleration Library (Intel® DAAL) version 2019 or later (https://github.com/01org/daal)
  * You can use the pre-built conda package from Intel channel on anaconda.org (see below)
* MPI
  * You can use the pre-built conda package intel or conda-forge channel on anaconda.org (see below)
  * Needed for distributed mode. You can disable support for distributed mode by setting NO_DIST to '1' or 'yes'

## Setting up a build environment
The easiest path for getting cython, DAAL, TBB, MPI etc. is by creating a conda environment and setting environment variables:
```
conda create -n DAAL4PY -c intel python=3.6 impi-devel tbb-devel daal daal-include cython jinja2 numpy
conda activate DAAL4PY
export TBBROOT=$CONDA_PREFIX
export DAALROOT=$CONDA_PREFIX
export MPIROOT=$CONDA_PREFIX
```

## Configuring the build with environment variables
* DAAL4PY_VERSION: sets package version
* NO_DIST: set to '1', 'yes' or alike to build without support for distributed mode
* NO_STREAM: set to '1', 'yes' or alike to build without support for streaming mode

### Notes on building for macOS
If building in High Sierra or higher, one may have to run into C++ build errors related to platform targets. Utilize ```export MACOSX_DEPLOYMENT_TARGET="10.9"``` if running into platform target issues.

## Building daal4py
Requires Intel® DAAL, Intel® TBB and MPI being properly setup, e.g. DAALROOT, TBBROOT and MPIROOT being set.
```
cd <checkout-dir>
python setup.py build_ext
```

## Installing daal4py
Requires Intel® DAAL, Intel® TBB and MPI being properly setup, e.g. DAALROOT, TBBROOT and MPIROOT being set.
```
cd <checkout-dir>
python setup.py install
```

# Building documentation for daal4py
## Prerequisites for creating documentation
* sphinx
* sphinx_rtd_theme

## Building documentation
1. Install daal4py into your python environment
2. ```cd doc && make html```
3. The documentation will be in ```doc/_build/html```
