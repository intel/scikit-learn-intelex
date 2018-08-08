[![Build Status](https://travis-ci.com/IntelPython/daal4py.svg?branch=master)](https://travis-ci.com/IntelPython/daal4py)
# daal4py - Convenient Python API to Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL)

With this API your Python programs can use Intel(R) DAAL algorithms in just one line:
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
* Intel(R) Concurrent Collections (Intel(R) CnC) version 1.2.300 or later (https://github.com/icnc/icnc)
  * You can use the pre-built conda package from Intel's test channel on anaconda.org (see below)
* Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) version 2018.0.3 or later (https://github.com/01org/daal)
  * You can use the pre-built conda package from Intel channel on anaconda.org (see below)
* MPICH
  * You can use the pre-built conda package conda-forge channel on anaconda.org (see below)

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

## VARIABLES
* DAAL4PY_VERSION: package version

## Building documentation
1. Install daal4py into your python environment
2. ```cd doc && make html```
3. The documentation will be in ```doc/_build/html```
