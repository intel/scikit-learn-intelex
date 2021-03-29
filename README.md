# Intel(R) Extension for Scikit-learn*
[![Build Status](https://dev.azure.com/daal/daal4py/_apis/build/status/CI?branchName=master)](https://dev.azure.com/daal/daal4py/_build/latest?definitionId=9&branchName=master)
[![Join the community on GitHub Discussions](https://badgen.net/badge/join%20the%20discussion/on%20github/black?icon=github)](https://github.com/intel/scikit-learn-intelex/discussions)

Intel(R) Extension for Scikit-learn speeds up scikit-learn beyond  by providing drop-in patching. Acceleration is achieved through the use of the Intel(R) oneAPI Data Analytics Library ([oneDAL](https://github.com/oneapi-src/oneDAL)) that allows for fast usage of the framework suited for Data Scientists or Machine Learning users.

⚠️Intel(R) Extension for Scikit-learn contains scikit-learn patching functionality originally available in [**daal4py**](/daal4py) package. All future updates for the patching will be available in Intel(R) Extension for Scikit-learn only. Please use the package instead of daal4py.

Running full the latest scikit-learn test suite with Intel(R) Extension for Scikit-learn: [![CircleCI](https://circleci.com/gh/intel/scikit-learn-intelex.svg?style=svg)](https://circleci.com/gh/intel/scikit-learn-intelex)

## 👀 Follow us on Medium

We publish blogs on Medium, so [follow us](https://medium.com/intel-analytics-software/tagged/machine-learning) to learn tips and tricks for more efficient data analysis the help of Intel(R) Extension for Scikit-learn. Here are our latest blogs:

- [From Hours to Minutes: 600x Faster SVM](https://medium.com/intel-analytics-software/from-hours-to-minutes-600x-faster-svm-647f904c31ae)
- [Improve the Performance of XGBoost and LightGBM Inference](https://medium.com/intel-analytics-software/improving-the-performance-of-xgboost-and-lightgbm-inference-3b542c03447e)
- [Accelerate Kaggle Challenges Using Intel AI Analytics Toolkit](https://medium.com/intel-analytics-software/accelerate-kaggle-challenges-using-intel-ai-analytics-toolkit-beb148f66d5a)
- [Accelerate Your scikit-learn Applications](https://medium.com/intel-analytics-software/improving-the-performance-of-xgboost-and-lightgbm-inference-3b542c03447e)
- [Accelerate Linear Models for Machine Learning](https://medium.com/intel-analytics-software/accelerating-linear-models-for-machine-learning-5a75ff50a0fe)
- [Accelerate K-Means Clustering](https://medium.com/intel-analytics-software/accelerate-k-means-clustering-6385088788a1)

## 🔗 Important links
- [Documentation](https://intel.github.io/scikit-learn-intelex/)
- [scikit-learn API and patching](https://intel.github.io/scikit-learn-intelex/)
- [Building from Sources](https://github.com/intel/scikit-learn-intelex/blob/master/INSTALL.md)
- [About Intel(R) oneAPI Data Analytics Library](https://github.com/oneapi-src/oneDAL)

## 💬 Support

Report issues, ask questions, and provide suggestions using:

- [GitHub Issues](https://github.com/intel/scikit-learn-intelex/issues)
- [GitHub Discussions](https://github.com/intel/scikit-learn-intelex/discussions)
- [Forum](https://community.intel.com/t5/Intel-Distribution-for-Python/bd-p/distribution-python)

You may reach out to project maintainers privately at onedal.maintainers@intel.com

# 🛠 Installation
Intel(R) Extension for Scikit-learn is available at the [Python Package Index](https://pypi.org/project/scikit-learn-intelex/),
and in [Intel channel](https://anaconda.org/intel/scikit-learn-intelex).

```bash
# PyPi (recommended by default)
pip install scikit-learn-intelex 
```

```bash
# Anaconda Cloud from Intel channel (recommended for Intel® Distribution for Python users)
conda install scikit-learn-intelex -c intel
```

<details><summary>[Click to expand] ℹ️ Supported configurations </summary>

#### 📦 PyPi channel

| OS / Python version     | **Python 3.6** | **Python 3.7** | **Python 3.8**| **Python 3.9**|
| :-----------------------| :------------: | :-------------:| :------------:| :------------:|
|    **Linux**            |    [CPU, GPU]  |  [CPU, GPU]    |   [CPU, GPU]  |  [CPU, GPU]|  |
|    **Windows**          |    [CPU, GPU]  |  [CPU, GPU]    |   [CPU, GPU]  |  [CPU, GPU]|  |
|    **OsX**              |    [CPU]       |  [CPU]         |    [CPU]      |    [CPU]      |

#### 📦 Anaconda Cloud: Intel channel

| OS / Python version     | **Python 3.6** | **Python 3.7** | **Python 3.8**| **Python 3.9**|
| :-----------------------| :------------: | :-------------:| :------------:| :------------:|
|    **Linux**            |   ❌          |     [CPU, GPU]  |     ❌       |      ❌       |
|    **Windows**          |   ❌          |     [CPU, GPU]  |     ❌       |      ❌       |
|    **OsX**              |   ❌          |     [CPU]       |     ❌       |      ❌       |

</details>

You can [build the package from sources](https://github.com/intel/scikit-learn-intelex/blob/master/INSTALL.md) as well.

# ⚡️ Get Started

Intel CPU optimizations patching
```py
import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.cluster import DBSCAN

X = np.array([[1., 2.], [2., 2.], [2., 3.],
              [8., 7.], [8., 8.], [25., 80.]], dtype=np.float32)
clustering = DBSCAN(eps=3, min_samples=2).fit(X)
```

Intel GPU optimizations patching
```py
import numpy as np
from sklearnex import patch_sklearn
from daal4py.oneapi import sycl_context
patch_sklearn()

from sklearn.cluster import DBSCAN

X = np.array([[1., 2.], [2., 2.], [2., 3.],
              [8., 7.], [8., 8.], [25., 80.]], dtype=np.float32)
with sycl_context("gpu"):
    clustering = DBSCAN(eps=3, min_samples=2).fit(X)
```

# 🚀 Scikit-learn patching

| Speedups of Intel(R) Extension for Scikit-learn over the original Scikit-learn |
|:--:|
| ![](https://github.com/intel/scikit-learn-intelex/blob/master/doc/IDP%20scikit-learn%20accelearation%20compared%20with%20stock%20scikit-learn.png) |
| *Technical details: float type: float64; HW: Intel(R) Xeon(R) Platinum 8280 CPU @ 2.70GHz, 2 sockets, 28 cores per socket; SW: scikit-learn 0.23.1, Intel® oneDAl (2021.1 Beta 10)* |

Intel(R) Extension for Scikit-learn patching affects performance of specific Scikit-learn functionality listed below. In cases when unsupported parameters are used, the package fallbacks into original Scikit-learn. These limitations described below. If the patching does not cover your scenarios, [submit an issue on GitHub](https://github.com/intel/scikit-learn-intelex/issues).

<details><summary>[Click to expand] 🔥 Applying the patching will impact the following existing scikit-learn algorithms: </summary>

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
||**roc_auc_score**|Parameters `average`, `sample_weight`, `max_fpr` and `multi_class` are not supported. | No limitations. |

 </details>

⚠️ We support optimizations for the last four versions of scikit-learn. The latest release of Intel(R) Extension for Scikit-learn 2021.2 supports scikit-learn 0.21.X,
0.22.X, 0.23.X and 0.24.X.

## 📜 Intel(R) Extension for Scikit-learn verbose

To find out which implementation of the algorithm is currently used (Intel(R) Extension for Scikit-learn or original Scikit-learn), set the environment variable:
- On Linux and Mac OS: `export SKLEARNEX_VERBOSE=INFO`
- On Windows: `set SKLEARNEX_VERBOSE=INFO`

For example, for DBSCAN you get one of these print statements depending on which implementation is used:
- `INFO: sklearn.cluster.DBSCAN.fit: uses Intel(R) oneAPI Data Analytics Library solver`
- `INFO: sklearn.cluster.DBSCAN.fit: uses original Scikit-learn solver`

[Read more in the documentation](https://intel.github.io/scikit-learn-intelex/).
