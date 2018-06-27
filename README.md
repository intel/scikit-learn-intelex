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

Please see GettingStartedHLDAAL.pdf for details.

# Building packages
## VARIABLES
* DAAL4PY_VERSION: package version
* R: path to R [R only]

## OVERVIEW
The build-process (using setup.py) is 2-phased
1.	Creating sources from C++ headers and preparing for package build
    * Note: You will not usually need this step because the repository contains the generated sources.
2.	Building the binary package

## PREREQUISITES
### Prerequisites for building binary packages
Below is the list of dependences

#### Linux
* Python 3.6
* jinja2
* cython
* A C++ compiler with C++11 support
* Intel(R) Threading Building Blocks (Intel(R) TBB) (https://www.threadingbuildingblocks.org/)
* Intel(R) Concurrent Collections (Intel(R) CnC) version 1.2.200 or later (https://github.com/icnc/icnc)
  * for python conda can automatically use the pre-built package from anaconda cloud (see below)
* Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) version 2018 (https://github.com/01org/daal)

#### Intel(R) CnC
*Note: When using conda, the pre-built package from Intel's test channel on anaconda.org (-c intel/label/test) is available (either online or for download). You need to install/build CnC only if you do not use conda - like when building interfaces for R.*
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
