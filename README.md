# daal4py - Convenient Python API to Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL)

**_This is a technical preview, not a product. Intel might decide to discontinue this project at any time._**

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
* jinja2
* cython
* numpy
* A C++ compiler with C++11 support
* Intel(R) Threading Building Blocks (Intel(R) TBB) (https://www.threadingbuildingblocks.org/)
* Intel(R) Concurrent Collections (Intel(R) CnC) version 1.2.200 or later (https://github.com/icnc/icnc)
  * for python conda can automatically use the pre-built package from anaconda cloud (see below)
* Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) version 2018 (https://github.com/01org/daal)
* Intel(R) MPI

### Setting up a build environment
The easiest path for getting cython, DAAL, TBB, CnC etc. is by creating a conda environment and setting environment variables:
```
conda create -n DAAL4PY -c intel -c intel/label/test python=3.6 mpi4py cnc tbb-devel daal daal-include cython jinja2 numpy
conda activate DAAL4PY
export CNCROOT=$CONDA_PREFIX
export TBBROOT=$CONDA_PREFIX
export DAALROOT=$CONDA_PREFIX
```

#### Intel(R) CnC
*Note: When using conda, the pre-built package from Intel's test channel on anaconda.org (-c intel/label/test) is available (either online or for download).
```
CNC_VERSION=1.2.200
git clone https://github.com/icnc/icnc v$CNC_VERSION
python make_kit.py -r $CNC_VERSION --mpi=<root-of-intel-mpi-install>/intel64 --itac=NONE
```
This creates a ready-to-use CnC install at kit.pkg/cnc/1.2.200
For more details please visit https://github.com/icnc/icnc.

## BUILDING DAAL4PY
Requires Intel(R) DAAL, Intel(R) TBB and Intel(R) CnC being properly setup, e.g. DAALROOT, TBBROOT and CNCROOT being set.
```
python setup.py build_ext
```

## INSTALLING DAAL4PY
Requires Intel(R) DAAL, Intel(R) TBB and Intel(R) CnC being properly setup, e.g. DAALROOT, TBBROOT and CNCROOT being set.
```
python setup.py install
```

## VARIABLES
* DAAL4PY_VERSION: package version
