# daal4py - A Convenient Python API to the Intel(R) oneAPI Data Analytics Library
[![Build Status](https://dev.azure.com/daal/daal4py/_apis/build/status/CI?branchName=master)](https://dev.azure.com/daal/daal4py/_build/latest?definitionId=9&branchName=master)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/daal4py.svg)](https://anaconda.org/conda-forge/daal4py)
[![Conda Version](https://anaconda.org/intel/daal4py/badges/version.svg)](https://anaconda.org/intel/daal4py)
[![Join the community on GitHub Discussions](https://badgen.net/badge/join%20the%20discussion/on%20github/black?icon=github)](https://github.com/IntelPython/daal4py/discussions)

A simplified API to Intel(R) oneAPI Data Analytics Library that allows for fast usage of the framework suited for Data Scientists or Machine Learning users.  Built to help provide an abstraction to Intel(R) oneAPI Data Analytics Library for either direct usage or integration into one's own framework and extending this beyond by providing drop-in paching for scikit-learn.

Running full scikit-learn test suite with daal4py optimization patches:

- [![CircleCI](https://circleci.com/gh/IntelPython/daal4py.svg?style=svg)](https://circleci.com/gh/IntelPython/daal4py) when applied to scikit-learn from PyPi
- [![CircleCI](https://circleci.com/gh/IntelPython/daal4py/tree/test-sklearn-master.svg?style=svg)](https://circleci.com/gh/IntelPython/daal4py/tree/test-sklearn-master) when applied to build from master branch

## Follow us on Medium

We publish blogs on Medium, so [follow us](https://medium.com/intel-analytics-software/tagged/machine-learning) to learn tips and tricks for more efficient data analysis the help of daal4py. Here are our latest blogs:

- [From Hours to Minutes: 600x Faster SVM](https://medium.com/intel-analytics-software/from-hours-to-minutes-600x-faster-svm-647f904c31ae)
- [Improve the Performance of XGBoost and LightGBM Inference](https://medium.com/intel-analytics-software/improving-the-performance-of-xgboost-and-lightgbm-inference-3b542c03447e)
- [Accelerate Kaggle Challenges Using Intel AI Analytics Toolkit](https://medium.com/intel-analytics-software/accelerate-kaggle-challenges-using-intel-ai-analytics-toolkit-beb148f66d5a)
- [Accelerate Your scikit-learn Applications](https://medium.com/intel-analytics-software/improving-the-performance-of-xgboost-and-lightgbm-inference-3b542c03447e)
- [Accelerate Linear Models for Machine Learning](https://medium.com/intel-analytics-software/accelerating-linear-models-for-machine-learning-5a75ff50a0fe)
- [Accelerate K-Means Clustering](https://medium.com/intel-analytics-software/accelerate-k-means-clustering-6385088788a1)

## Important links
- [Documentation](https://intelpython.github.io/daal4py/)
- [scikit-learn API and patching](https://intelpython.github.io/daal4py/sklearn.html#sklearn)
- [Source Code](https://github.com/IntelPython/daal4py/tree/master/src)
- [Building from Sources](INSTALL.md)
- [About Intel(R) oneAPI Data Analytics Library](https://oneapi-src.github.io/oneDAL/)

## Support

Report issues, ask questions, and provide suggestions using:

- [GitHub Issues](https://github.com/IntelPython/daal4py/issues)
- [GitHub Discussions](https://github.com/IntelPython/daal4py/discussions)
- [Forum](https://community.intel.com/t5/Intel-Distribution-for-Python/bd-p/distribution-python)

You may reach out to project maintainers privately at onedal.maintainers@intel.com

# Installation
daal4py can be installed from conda-forge (recommended):
```bash
conda install daal4py -c conda-forge
```
or from Intel channel:
```bash
conda install daal4py -c intel
```
You can [build daal4py from sources](INSTALL.md) as well.

# Getting Started
Core functioanlity of daal4py is in place Scikit-learn patching - Same Code, Same Behavior but faster execution. 

Intel CPU optimizations patching
```py
import numpy as np
from daal4py.sklearn import patch_sklearn
patch_sklearn()

from sklearn.cluster import DBSCAN

X = np.array([[1., 2.], [2., 2.], [2., 3.],
              [8., 7.], [8., 8.], [25., 80.]], dtype=np.float32)
clustering = DBSCAN(eps=3, min_samples=2).fit(X)
```

Intel CPU/GPU optimizations patching
```py
import numpy as np
from daal4py.sklearn import patch_sklearn
from daal4py.oneapi import sycl_context
patch_sklearn()

from sklearn.cluster import DBSCAN

X = np.array([[1., 2.], [2., 2.], [2., 3.],
              [8., 7.], [8., 8.], [25., 80.]], dtype=np.float32)
with sycl_context("gpu"):
    clustering = DBSCAN(eps=3, min_samples=2).fit(X)
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

| *Speedups of daal4py-powered Scikit-learn over the original Scikit-learn* |
|:--:|
| ![](doc/IDP%20scikit-learn%20accelearation%20compared%20with%20stock%20scikit-learn.png) |
| *Technical details: float type: float64; HW: Intel(R) Xeon(R) Platinum 8280 CPU @ 2.70GHz, 2 sockets, 28 cores per socket; SW: scikit-learn 0.23.1, Intel® oneDAl (2021.1 Beta 10)* |

daal4py patching affects performance of specific Scikit-learn functionality listed below. In cases when unsupported parameters are used, daal4py fallbacks into stock Scikit-learn. These limitations described below. If the patching does not cover your scenarios, [submit an issue on GitHub](https://github.com/IntelPython/daal4py/issues).

Scenarios that are already available in 2020.3 release:
|Task|Functionality|Parameters support|Data support|
|:---|:------------|:-----------------|:-----------|
|Classification|**SVC**|All parameters except `kernel` = 'poly' and 'sigmoid'. | No limitations.|
||**RandomForestClassifier**|All parameters except `warmstart` = True and `cpp_alpha` != 0, `criterion` != 'gini'. | Multi-output and sparse data is not supported. |
||**KNeighborsClassifier**|All parameters except `metric` != 'euclidean' or `minkowski` with `p` = 2. | Multi-output and sparse data is not supported. |
||**LogisticRegression / LogisticRegressionCV**|All parameters except `solver` != 'lbfgs' or 'newton-cg', `class_weight` != None, `sample_weight` != None. | Only dense data is supported. |
|Regression|**RandomForestRegressor**|All parameters except `warmstart` = True and `cpp_alpha` != 0, `criterion` != 'mse'. | Multi-output and sparse data is not supported. |
||**KNeighborsRegressor**|All parameters except `metric` != 'euclidean' or `minkowski` with `p` = 2. | Sparse data is not supported. |
||**LinearRegression**|All parameters except `normalize` != False and `sample_weight` != None. | Only dense data is supported, `#observations` should be >= `#features`. |
||**Ridge**|All parameters except `normalize` != False, `solver` != 'auto' and `sample_weight` != None. | Only dense data is supported, `#observations` should be >= `#features`. |
||**ElasticNet**|All parameters except `sample_weight` != None. | Multi-output and sparse data is not supported, `#observations` should be >= `#features`. |
||**Lasso**|All parameters except `sample_weight` != None. | Multi-output and sparse data is not supported, `#observations` should be >= `#features`. |
|Clustering|**KMeans**|All parameters except `precompute_distances` and `sample_weight` != None. | No limitations. |
||**DBSCAN**|All parameters except `metric` != 'euclidean' or `minkowski` with `p` = 2. | Only dense data is supported. |
|Dimensionality reduction|**PCA**|All parameters except `svd_solver` != 'full'. | No limitations. |
|| **TSNE**|All parameters except `metric` != 'euclidean' or `minkowski` with `p` = 2. | Sparse data is not supported. |
|Unsupervised|**NearestNeighbors**|All parameters except `metric` != 'euclidean' or `minkowski` with `p` = 2. | Sparse data is not supported. |
|Other|**train_test_split**|All parameters are supported. | Only dense data is supported.|
||**assert_all_finite**|All parameters are supported. | Only dense data is supported. |
||**pairwise_distance**|With `metric`='cosine' and 'correlation'.| Only dense data is supported. |

Scenarios that are only available in the `master` branch (not released yet):

|Task|Functionality|Parameters support|Data support|
|:---|:------------|:-----------------|:-----------|
|Other|**roc_auc_score**|Parameters `average`, `sample_weight`, `max_fpr` and `multi_class` are not supported. | No limitations. |

## scikit-learn verbose

To find out which implementation of the algorithm is currently used (daal4py or stock Scikit-learn), set the environment variable:
- On Linux and Mac OS: `export IDP_SKLEARN_VERBOSE=INFO`
- On Windows: `set IDP_SKLEARN_VERBOSE=INFO`

For example, for DBSCAN you get one of these print statements depending on which implementation is used:
- `INFO: sklearn.cluster.DBSCAN.fit: uses Intel(R) oneAPI Data Analytics Library solver`
- `INFO: sklearn.cluster.DBSCAN.fit: uses original Scikit-learn solver`

[Read more in the documentation](https://intelpython.github.io/daal4py/sklearn.html#scikit-learn-verbose).

# Building from Source
See [Building from Sources](INSTALL.md) for details.
