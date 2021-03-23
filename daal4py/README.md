# daal4py - A Convenient Python API to the Intel(R) oneAPI Data Analytics Library
[![Build Status](https://dev.azure.com/daal/daal4py/_apis/build/status/CI?branchName=master)](https://dev.azure.com/daal/daal4py/_build/latest?definitionId=9&branchName=master)
[![Join the community on GitHub Discussions](https://badgen.net/badge/join%20the%20discussion/on%20github/black?icon=github)](https://github.com/IntelPython/daal4py/discussions)
[![PyPI Version](https://img.shields.io/pypi/v/daal4py)](https://pypi.org/project/daal4py/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/daal4py)](https://anaconda.org/conda-forge/daal4py)


A simplified API to Intel(R) oneAPI Data Analytics Library that allows for fast usage of the framework suited for Data Scientists or Machine Learning users.  Built to help provide an abstraction to Intel(R) oneAPI Data Analytics Library for either direct usage or integration into one's own framework.

## 👀 Follow us on Medium

We publish blogs on Medium, so [follow us](https://medium.com/intel-analytics-software/tagged/machine-learning) to learn tips and tricks for more efficient data analysis the help of daal4py. Here are our latest blogs:

- [From Hours to Minutes: 600x Faster SVM](https://medium.com/intel-analytics-software/from-hours-to-minutes-600x-faster-svm-647f904c31ae)
- [Improve the Performance of XGBoost and LightGBM Inference](https://medium.com/intel-analytics-software/improving-the-performance-of-xgboost-and-lightgbm-inference-3b542c03447e)
- [Accelerate Kaggle Challenges Using Intel AI Analytics Toolkit](https://medium.com/intel-analytics-software/accelerate-kaggle-challenges-using-intel-ai-analytics-toolkit-beb148f66d5a)
- [Accelerate Your scikit-learn Applications](https://medium.com/intel-analytics-software/improving-the-performance-of-xgboost-and-lightgbm-inference-3b542c03447e)
- [Accelerate Linear Models for Machine Learning](https://medium.com/intel-analytics-software/accelerating-linear-models-for-machine-learning-5a75ff50a0fe)
- [Accelerate K-Means Clustering](https://medium.com/intel-analytics-software/accelerate-k-means-clustering-6385088788a1)

## 🔗 Important links
- [Documentation](https://intelpython.github.io/daal4py/)
- [scikit-learn API and patching](https://intelpython.github.io/daal4py/sklearn.html#sklearn)
- [Building from Sources](https://github.com/IntelPython/daal4py/blob/master/daal4y/INSTALL.md)
- [About Intel(R) oneAPI Data Analytics Library](https://github.com/oneapi-src/oneDAL)

## 💬 Support

Report issues, ask questions, and provide suggestions using:

- [GitHub Issues](https://github.com/IntelPython/daal4py/issues)
- [GitHub Discussions](https://github.com/IntelPython/daal4py/discussions)
- [Forum](https://community.intel.com/t5/Intel-Distribution-for-Python/bd-p/distribution-python)

You may reach out to project maintainers privately at onedal.maintainers@intel.com

# 🛠 Installation
daal4py is available at the [Python Package Index](https://pypi.org/project/daal4py/),
on Anaconda Cloud in [Conda-Forge channel](https://anaconda.org/conda-forge/daal4py)
and in [Intel channel](https://anaconda.org/intel/daal4py).

```bash
# PyPi
pip install daal4py (recommended for PyPI users)
```

```bash
# Anaconda Cloud from Conda-Forge channel (recommended for conda users by default)
conda install daal4py -c conda-forge
```

```bash
# Anaconda Cloud from Intel channel (recommended for Intel® Distribution for Python users)
conda install daal4py -c intel
```

<details><summary>[Click to expand] ℹ️ Supported configurations </summary>

#### 📦 PyPi channel

| OS / Python version     | **Python 3.6** | **Python 3.7** | **Python 3.8**| **Python 3.9**|
| :-----------------------| :------------: | :-------------:| :------------:| :------------:|
|    **Linux**            |    [CPU, GPU]  |  [CPU, GPU]    |   [CPU, GPU]  |  [CPU, GPU]|  |
|    **Windows**          |    [CPU, GPU]  |  [CPU, GPU]    |   [CPU, GPU]  |  [CPU, GPU]|  |
|    **OsX**              |    [CPU]       |  [CPU]         |    [CPU]      |    [CPU]      |

#### 📦 Anaconda Cloud: Conda-Forge channel

| OS / Python version     | **Python 3.6** | **Python 3.7** | **Python 3.8**| **Python 3.9**|
| :-----------------------| :------------: | :------------: | :------------:| :------------:|
|    **Linux**            |   [CPU]        |   [CPU]        |     [CPU]     |     [CPU]     |
|    **Windows**          |   [CPU]        |   [CPU]        |     [CPU]     |     [CPU]     |
|    **OsX**              |   ❌           |     ❌        |     ❌        |       ❌     |

#### 📦 Anaconda Cloud: Intel channel

| OS / Python version     | **Python 3.6** | **Python 3.7** | **Python 3.8**| **Python 3.9**|
| :-----------------------| :------------: | :-------------:| :------------:| :------------:|
|    **Linux**            |   ❌          |     [CPU, GPU]  |     ❌       |      ❌       |
|    **Windows**          |   ❌          |     [CPU, GPU]  |     ❌       |      ❌       |
|    **OsX**              |   ❌          |     [CPU]       |     ❌       |      ❌       |

</details>

You can [build daal4py from sources](https://github.com/IntelPython/daal4py/blob/master/daal4py/INSTALL.md) as well.


# 🚀 Scikit-learn patching

Scikit-learn patching functionality in daal4py was deprecated and moved to a separate package - Intel(R) Extension for Scikit-learn*. All future updates for the patching will be available in Intel(R) Extension for Scikit-learn only. Please use the package instead of daal4py for the Scikit-learn acceleration.
