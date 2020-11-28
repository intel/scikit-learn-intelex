# daal4py - A Convenient Python API to the Intel(R) oneAPI Data Analytics Library
[![Build Status](https://travis-ci.com/IntelPython/daal4py.svg?branch=master)](https://travis-ci.com/IntelPython/daal4py)
[![Build Status](https://dev.azure.com/frankschlimbach/daal4py/_apis/build/status/IntelPython.daal4py?branchName=master)](https://dev.azure.com/frankschlimbach/daal4py/_build/latest?definitionId=1&branchName=master)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/daal4py.svg)](https://anaconda.org/conda-forge/daal4py)
[![Conda Version](https://anaconda.org/intel/daal4py/badges/version.svg)](https://anaconda.org/intel/daal4py)

A simplified API to Intel(R) oneAPI Data Analytics Library that allows for fast usage of the framework suited for Data Scientists or Machine Learning users.  Built to help provide an abstraction to Intel(R) oneAPI Data Analytics Library for either direct usage or integration into one's own framework and extending this beyond by providing drop-in paching for scikit-learn.

- [Documentation](https://intelpython.github.io/daal4py/)
- [scikit-learn API and patching](https://intelpython.github.io/daal4py/sklearn.html#sklearn)
- [Source Code](https://github.com/IntelPython/daal4py/tree/master/src)
- [About Intel(R) oneAPI Data Analytics Library](https://oneapi-src.github.io/oneDAL/)

Running full scikit-learn test suite with daal4p's optimization patches

- [![CircleCI](https://circleci.com/gh/IntelPython/daal4py.svg?style=svg)](https://circleci.com/gh/IntelPython/daal4py) when applied to scikit-learn from PyPi
- [![CircleCI](https://circleci.com/gh/IntelPython/daal4py/tree/test-sklearn-master.svg?style=svg)](https://circleci.com/gh/IntelPython/daal4py/tree/test-sklearn-master) when applied to build from master branch

# Installation
daal4py can be installed from conda-forge (recommended):
```bash
conda install daal4py -c conda-forge
```
or from Intel channel:
```bash
conda install daal4py -c intel
```

# Getting Started
Core functioanlity of daal4py is in place Scikit-learn patching - Same Code, Same Behavior but faster execution. 

Intel CPU optimizations patching
```py
from daal4py.sklearn import patch_sklearn
patch_sklearn()

from sklearn.svm import SVC
from sklearn.datasets import load_digits
digits = load_digits()
X, y = digits.data, digits.target
clf = SVC().fit(X, y)
res = clf.predict(X)
```

Intel CPU/GPU optimizations patching
```py
from daal4py.sklearn import patch_sklearn
from daal4py.oneapi import sycl_context
patch_sklearn()

from sklearn.svm import SVC
from sklearn.datasets import load_digits
digits = load_digits()
X, y = digits.data, digits.target
with sycl_context("gpu"):
    clf = SVC().fit(X, y)
    res = clf.predict(X)
```
daal4py API, allows you to use wider set of Intel(R) oneAPI Data Analytics Library algorithms in just one line:
```py
import daal4py as d4p
init = d4p.kmeans_init(data, 10, t_method="plusPlusDense")
result = init.compute(X)
```
You can even run this on a cluster by making simple code changes:
```py
import daal4py as d4p
d4p.daalinit()
d4p.kmeans_init(data, 10, t_method="plusPlusDense", distributed=True)
result = init.compute(X, daal4py.my_procid())
d4p.daalfini()
```

# Scikit-learn patching

| *Speedups of daal4py-powered Scikit-learn over the original Scikit-learn, 28 cores, 1 thread/core* |
|:--:|
| ![](doc/IDP%20scikit-learn%20accelearation%20compared%20with%20stock%20scikit-learn.png) |
| *technical details: FPType: float32; HW: Intel(R) Xeon(R) Platinum 8276L CPU @ 2.20GHz, 2 sockets, 28 cores per socket; SW: scikit-learn 0.22.2, Intel® DAAL (2019.5), Intel® Distribution Of Python (IDP) 3.7.4; Details available in [the related article on Medium](https://medium.com/intel-analytics-software/accelerate-your-scikit-learn-applications-a06cacf44912)* |

daal4py patching affects performance of specific Scikit-learn functionality listed below. In cases when unsupported parameters are used, daal4py fallbacks into stock Scikit-learn. These limitations described below. If the patching does not cover your scenarios, [submit an issue on GitHub](https://github.com/IntelPython/daal4py/issues).

Scenarios that are already available in 2020.3 release:
|Task|Functionality|Parameters support|Data support|
|:---|:------------|:-----------------|:-----------|
|Classification|**SVC**|All parameters except `kernel` = 'poly' and 'sigmoid'. | No limitations.|
||**RandomForestClassifier**|All parameters except `warmstart` = True and `cpp_alpha` != 0, `criterion` != 'gini'. | Multi-output and sparse data is not supported. |
||**KNeighborsClassifier**|Supported `metric` = 'euclidean' and `minkowski` with `p` = 2. Other parameters are fully supported. | Multi-output and sparse data is not supported. |
||**LogisticRegression / LogisticRegressionCV**|Supported `solver` = 'lbfgs' and 'newton-cg', `penalty` = 'l2' and 'none', `class_weight` = None. Other parameters are fully supported. | Only dense data is supported. |
|Regression|**RandomForestRegressor**|All parameters except `warmstart` = True and `cpp_alpha` != 0, `criterion` != 'mse'. | Multi-output and sparse data is not supported. |
||**LinearRegression**|All parameters except `normalize` != False. | Only dense data is supported, `#observations` should be >= `#features`. |
||**Ridge**|All parameters except `normalize` != False and `solver` != 'auto'. | Only dense data is supported, `#observations` should be >= `#features`. |
||**ElasticNet**|All parameters are supported| Multi-output and sparse data is not supported, `#observations` should be >= `#features`. |
||**Lasso**|All parameters are supported| Multi-output and sparse data is not supported, `#observations` should be >= `#features`. |
|Clustering|**KMeans**|All parameters except `precompute_distances`. | No limitations. |
||**DBSCAN**|Supported `metric` = 'euclidean' and 'minkowski' with p='2', `algorithm`='brute. Other parameters are fully supported. | Only dense data is supported. |
|Dimensionality reduction|**PCA**|All parameters except `svd_solver` != 'full'. | No limitations. |
|Other|**train_test_split**|All parameters are supported. | Only dense data is supported.|
||**assert_all_finite**|All parameters are supported. | Only dense data is supported. |
||**pairwise_distance**|With `metric`='cosine' and 'correlation'.| Only dense data is supported. |

Scenarios that are only available in the `master` branch (not released yet):

|Task|Functionality|Parameters support|Data support|
|:---|:------------|:-----------------|:-----------|
|Regression|**KNeighborsRegressor**|Supported `metric` = 'euclidean' and 'minkowski' with `p` = 2. Other parameters are fully supported. | Sparse data is not supported. |
|Unsupervised|**NearestNeighbors**|Supported `metric` = 'euclidean' and 'minkowski' with `p` = 2. Other parameters are fully supported. | Sparse data is not supported. |
|Dimensionality reduction|**TSNE**|Supported `metric` = 'euclidean' and 'minkowski' with `p` = 2. Other parameters are fully supported. | Sparse data is not supported. |
|Other|**roc_auc_score**|Parameters `average`, `sample_weight`, `max_fpr` and `multi_class` are not supported. | No limitations. |


# Building from sources
daal4py is easily built from sources with the majority of the necessary prerequisites available on conda. The instructions below detail how to gather the prerequisites, set your build environment, and finally build and install the completed package. daal4py can be built for all three major platforms (Windows, Linux, macOS). Multi-node (distributed) and streaming support can be disabled if needed. 

The build-process (using setup.py) happens in 3 stages:
1. Creating C++ and cython sources from oneDAL C++ headers
2. Running cython on generated source
3. Compiling and linking

To build with oneAPI support, additional steps are required:
1. Point to DPC++ compiler by defining ``DPCPPROOT`` variable.

```bash
export DPCPPROOT=/opt/intel/oneapi/compiler/latest
```
2. Install Intel(R) oneAPI Data Analytics Library with oneAPI support:

    - From Conda channel.
    - From oneAPI packages repository (pass the path to oneDAL via ``DALROOT`` variable)

```bash
export DALROOT=/opt/intel/oneapi/daal/latest
```

# Building daal4py using conda-build
The easiest way to build daal4py is using the conda-build with the provided recipe.

## Prerequisites
* Python version >= 3.6
* conda-build version >= 3
* C++ compiler with C++11 support

For oneAPI support:
* A DPC++ compiler 
* Intel(R) oneAPI Data Analytics Library version 2021.1 or later (https://github.com/oneapi-src/oneDAL)
  * You can use the pre-built conda package from Intel channel on anaconda.org

## Building daal4py
Library build command:
```bash
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
conda install -c intel -c conda-forge mpich daal numpy
```
Windows:
```
conda install -c intel mpi_rt daal numpy
```

# Building daal4py without conda-build
Without conda-build you have to manually setup your environment before building daal4py.

## Prerequisites
* Python version >= 3.6
* Jinja2
* Cython
* Numpy
* A C++ compiler with C++11 support
* Intel® Data Analytics Acceleration Library (Intel® DAAL) version 2019 or later (https://github.com/01org/daal)
  * You can use the pre-built conda package from Intel channel on anaconda.org
* MPI
  * You can use the pre-built conda package intel or conda-forge channel on anaconda.org
  * Needed for distributed mode. You can disable support for distributed mode by setting NO_DIST to '1' or 'yes'
For oneAPI support
* A DPC++ compiler 
* Intel(R) oneAPI Data Analytics Library version 2021.1 or later (https://oneapi-src.github.io/oneDAL/)
  * You can use the pre-built conda package from Intel channel on anaconda.org

## Setting up a build environment
The easiest path for getting cython, oneDAL, MPI etc. is by creating a conda environment and setting environment variables:
```
conda create -n DAAL4PY python=3.7 impi-devel daal daal-include cython jinja2 numpy clang-tools -c intel -c conda-forge
conda activate DAAL4PY
export DALROOT=$CONDA_PREFIX
export MPIROOT=$CONDA_PREFIX
```

## Configuring the build with environment variables
* DAAL4PY_VERSION: sets package version
* NO_DIST: set to '1', 'yes' or alike to build without support for distributed mode
* NO_STREAM: set to '1', 'yes' or alike to build without support for streaming mode

### Notes on building for macOS
If building in High Sierra or higher, one may have to run into C++ build errors related to platform targets. Utilize ```export MACOSX_DEPLOYMENT_TARGET="10.9"``` if running into platform target issues.

## Building daal4py
Requires Intel(R) oneAPI Data Analytics Library and Intel(R) MPI Library being properly set up, meaning you have to set DALROOT and MPIROOT variables.
```bash
cd <checkout-dir>
python setup.py build_ext
```

## Installing daal4py
Requires Intel(R) oneAPI Data Analytics Library and Intel(R) MPI Library being properly set up, meaning you have to set DALROOT and MPIROOT variables.
```bash
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
