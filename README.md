# daal4py - A Convenient Python API to the Intel® Data Analytics Acceleration Library (Intel® DAAL)
[![Build Status](https://travis-ci.com/IntelPython/daal4py.svg?branch=master)](https://travis-ci.com/IntelPython/daal4py)

A simplified API to Intel® DAAL that allows for fast usage of the framework suited for Data Scientists or Machine Learning users.  Built to help provide an abstraction to Intel® DAAL for either direct usage or integration into one's own framework.  

- [Documentation](https://intelpython.github.io/daal4py/)
- [Source Code](https://github.com/IntelPython/daal4py/tree/master/src)
- [About Intel® DAAL](https://software.intel.com/en-us/intel-daal)

With this daal4py API, your Python programs can use Intel® DAAL algorithms in just one line:
```
kmeans_init(data, 10, t_method="plusPlusDense")
```
You can even run this on a cluster by simple adding a keyword-parameter
```
kmeans_init(data, 10, t_method="plusPlusDense", distributed=TRUE)
```

# Building packages


## OVERVIEW
The build-process (using setup.py) is 3-phased
1. Creating C++ and cython sources from DAAL C++ headers
2. Running cython on generated source
3. Compiling and linking

## PREREQUISITES
### Prerequisites for building binary packages
* Python 3.6
* Jinja2
* Cython
* Numpy
* A C++ compiler with C++11 support
* Intel(R) Threading Building Blocks (Intel(R) TBB) version 2018.0.4 or later (https://www.threadingbuildingblocks.org/)
  * You can use the pre-built conda package from Intel's channel or conda-forge channel on anaconda.org (see below)
  * Needed for distributed mode. You can disable support for distributed mode by setting NO_DIST to '1' or 'yes'
* Intel(R) Concurrent Collections (Intel(R) CnC) version 1.2.300 or later (https://github.com/icnc/icnc)
  * You can use the pre-built conda package from Intel's test channel on anaconda.org (see below)
* Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) version 2018.0.3 or later (https://github.com/01org/daal)
  * You can use the pre-built conda package from Intel channel on anaconda.org (see below)
* MPICH
  * You can use the pre-built conda package conda-forge channel on anaconda.org (see below)
  * Needed for distributed mode. You can disable support for distributed mode by setting NO_DIST to '1' or 'yes'

### Prerequisites for creating documentation
* sphinx
* sphinx_rtd_theme


### Setting up a build environment
The easiest path for getting cython, DAAL, TBB, CnC etc. is by creating a conda environment and setting environment variables:
```
conda create -n DAAL4PY -c intel -c intel/label/test -c conda-forge python=3.6 mpich cnc tbb-devel daal daal-include cython jinja2 numpy
conda activate DAAL4PY
export CNCROOT=$CONDA_PREFIX
export TBBROOT=$CONDA_PREFIX
export DAALROOT=$CONDA_PREFIX
```

### C++ compatibility
The CnC binary packages on the intel channel are compiled with g++ version 4.8. For using newer compilers, globally #define macro ```_GLIBCXX_USE_CXX11_ABI=0``` when compiling daal4py or compile CnC with the newer compiler.

## Building daal4py
Requires Intel(R) DAAL, Intel(R) TBB and Intel(R) CnC being properly setup, e.g. DAALROOT, TBBROOT and CNCROOT being set.
```
python setup.py build_ext
```

## Installing daal4py
Requires Intel(R) DAAL, Intel(R) TBB and Intel(R) CnC being properly setup, e.g. DAALROOT, TBBROOT and CNCROOT being set.
```
python setup.py install
```

## Configuring the build with environment variables
* DAAL4PY_VERSION: sets package version
* NO_DIST: set to '1', 'yes' or alike to build without support for distributed mode

## Building documentation
1. Install daal4py into your python environment
2. ```cd doc && make html```
3. The documentation will be in ```doc/_build/html```
