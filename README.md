# Intel(R) Extension for Scikit-learn* <img align="right" width="100" height="100" src="https://spec.oneapi.io/oneapi-logo-white-scaled.jpg">

[Installation](https://github.com/intel/scikit-learn-intelex#-installation)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Get Started](https://github.com/intel/scikit-learn-intelex#%EF%B8%8F-get-started)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Documentation](https://intel.github.io/scikit-learn-intelex/)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Examples](https://github.com/intel/scikit-learn-intelex/tree/master/examples/notebooks)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Medium blogs](https://medium.com/intel-analytics-software/tagged/machine-learning)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Support](https://github.com/intel/scikit-learn-intelex#-support)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[FAQ](#-faq)&nbsp;&nbsp;&nbsp;

[![Build Status](https://dev.azure.com/daal/daal4py/_apis/build/status/CI?branchName=master)](https://dev.azure.com/daal/daal4py/_build/latest?definitionId=9&branchName=master)
[![Coverity Scan Build Status](https://scan.coverity.com/projects/21716/badge.svg)](https://scan.coverity.com/projects/daal4py)
[![Join the community on GitHub Discussions](https://badgen.net/badge/join%20the%20discussion/on%20github/black?icon=github)](https://github.com/intel/scikit-learn-intelex/discussions)
[![PyPI Version](https://img.shields.io/pypi/v/scikit-learn-intelex)](https://pypi.org/project/scikit-learn-intelex/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/scikit-learn-intelex)](https://anaconda.org/conda-forge/scikit-learn-intelex)

With Intel(R) Extension for Scikit-learn you can speed up your Scikit-learn applications without changing the existing code.
Intel(R) Extension for Scikit-learn is part of [oneAPI](https://oneapi.io) and [Intel® oneAPI AI Analytics Toolkit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html) (AI Kit).

The status of the latest scikit-learn test suite with Intel(R) Extension for Scikit-learn: [![CircleCI](https://circleci.com/gh/intel/scikit-learn-intelex.svg?style=svg)](https://circleci.com/gh/intel/scikit-learn-intelex)

## How it works?

Intel(R) Extension for Scikit-learn offers you a way to accelerate existing scikit-learn code.
The acceleration is achieved through **patching**: you import an additional Python package and enable optimizations via `sklearnex.patch_sklearn()`. Jump to [Get Started](#️-get-started) section to see examples.

❗ The patching only affects [selected algorithms and their parameters](https://intel.github.io/scikit-learn-intelex/algorithms.html). 

You may still use algorithms and parameters not supported by Intel(R) Extension for Scikit-learn in your code. You will not get an error if you do this. When you use algorithms or parameters not supported by the extension, the package fallbacks into original stock version of scikit-learn.

## 🛠 Installation
Intel(R) Extension for Scikit-learn is available at the [Python Package Index](https://pypi.org/project/scikit-learn-intelex/),
on Anaconda Cloud in [Conda-Forge channel](https://anaconda.org/conda-forge/scikit-learn-intelex) and in [Intel channel](https://anaconda.org/intel/scikit-learn-intelex).
The extension is also available as a part of [Intel® oneAPI AI Analytics Toolkit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html) (AI Kit).

You can [build the `scikit-learn-intelex` package from sources](https://github.com/intel/scikit-learn-intelex/blob/master/INSTALL.md) as well.

Refer to 📦 [System Requirements](https://intel.github.io/scikit-learn-intelex/system_requirements.html) for details about hardware, operating system, and software prerequisites for Intel® Extension for Scikit-learn.

### 🔧 Install from PyPI channel (recommended by default)

```bash
pip install scikit-learn-intelex
```

#### 📦 Supported configurations for PyPI

| OS / Python version | **Python 3.6** | **Python 3.7** | **Python 3.8** | **Python 3.9** |
| :------------------ | :------------: | :------------: | :------------: | :------------: |
| **Linux**           |   [CPU, GPU]   |   [CPU, GPU]   |   [CPU, GPU]   |   [CPU, GPU]   |
| **Windows**         |   [CPU, GPU]   |   [CPU, GPU]   |   [CPU, GPU]   |   [CPU, GPU]   |
| **OsX**             |     [CPU]      |     [CPU]      |     [CPU]      |     [CPU]      |


### 🔧 Install from Anaconda Cloud

<details><summary>[See instructions and supported configurations]</summary>

#### 🔧 Install via Anaconda Cloud from Conda-Forge channel

```bash
conda install scikit-learn-intelex -c conda-forge
```

##### 📦 Supported configurations for  Anaconda Cloud from Conda-Forge channel

| OS / Python version     | **Python 3.6** | **Python 3.7** | **Python 3.8**| **Python 3.9**|
| :-----------------------| :------------: | :------------: | :------------:| :------------:|
|    **Linux**            |   [CPU]        |   [CPU]        |     [CPU]     |     [CPU]     |
|    **Windows**          |   [CPU]        |   [CPU]        |     [CPU]     |     [CPU]     |
|    **OsX**              |   [CPU]        |   [CPU]        |     [CPU]     |     [CPU]     |

#### 🔧 Install via Anaconda Cloud from Intel channel

We recommend this installation for the users of Intel® Distribution for Python.

```bash
conda install scikit-learn-intelex -c intel
```

##### 📦 Supported configurations for Anaconda Cloud from Intel channel

| OS / Python version | **Python 3.6** | **Python 3.7** | **Python 3.8** | **Python 3.9** |
| :------------------ | :------------: | :------------: | :------------: | :------------: |
| **Linux**           |   [CPU, GPU]   |   [CPU, GPU]   |   [CPU, GPU]   |   [CPU, GPU]   |
| **Windows**         |   [CPU, GPU]   |   [CPU, GPU]   |   [CPU, GPU]   |   [CPU, GPU]   |
| **OsX**             |     [CPU]      |     [CPU]      |     [CPU]      |     [CPU]      |

</details>

## ⚡️ Get Started

Check out available [notebooks](https://github.com/intel/scikit-learn-intelex/tree/master/examples/notebooks) for more examples.

### Enable Intel CPU optimizations
```py
import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.cluster import DBSCAN

X = np.array([[1., 2.], [2., 2.], [2., 3.],
              [8., 7.], [8., 8.], [25., 80.]], dtype=np.float32)
clustering = DBSCAN(eps=3, min_samples=2).fit(X)
```

### Enable Intel GPU optimizations
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

## ❔ FAQ

<details><summary>[See answers to frequently asked questions]</summary>

### ❓ Are all algorithms affected by patching?

> No. The patching only affects [selected algorithms and their parameters](https://intel.github.io/scikit-learn-intelex/algorithms.html). 

### ❓ What happens if I use parameters not supported by the extension?

> In cases when unsupported parameters are used, the package fallbacks into original stock version of scikit-learn. You will not get an error.

### ❓ What happens if I run algorithms not supported by the extension?

> If you use algorithms for which no optimizations are available, their original version from the stock scikit-learn is used.

### ❓ Can I see which implementation of the algorithm is currently used?

> Yes. To find out which implementation of the algorithm is currently used (Intel(R) Extension for Scikit-learn or original Scikit-learn), use the [verbose mode](https://intel.github.io/scikit-learn-intelex/verbose.html).

### ❓ How much faster scikit-learn is after the patching?

> We compare the performance of Intel(R) Extension for Scikit-Learn to other frameworks in [Machine Learning Benchmarks](https://github.com/IntelPython/scikit-learn_bench). Read [our blogs on Medium](#-follow-us-on-medium) if you are interested in the detailed comparison.

### ❓ What if the patching does not cover my scenario?

> If the patching does not cover your scenarios, [submit an issue on GitHub](https://github.com/intel/scikit-learn-intelex/issues) with the description of what you would want to have.

</details>

## 👀 Follow us on Medium

We publish blogs on Medium, so [follow us](https://medium.com/intel-analytics-software/tagged/machine-learning) to learn tips and tricks for more efficient data analysis with the help of Intel(R) Extension for Scikit-learn. Here are our latest blogs:

- [Save Time and Money with Intel Extension for Scikit-learn](https://medium.com/intel-analytics-software/save-time-and-money-with-intel-extension-for-scikit-learn-33627425ae4)
- [Superior Machine Learning Performance on the Latest Intel Xeon Scalable Processors](https://medium.com/intel-analytics-software/superior-machine-learning-performance-on-the-latest-intel-xeon-scalable-processor-efdec279f5a3)
- [Leverage Intel Optimizations in Scikit-Learn](https://medium.com/intel-analytics-software/leverage-intel-optimizations-in-scikit-learn-f562cb9d5544)
- [Intel Gives Scikit-Learn the Performance Boost Data Scientists Need](https://medium.com/intel-analytics-software/intel-gives-scikit-learn-the-performance-boost-data-scientists-need-42eb47c80b18)
- [From Hours to Minutes: 600x Faster SVM](https://medium.com/intel-analytics-software/from-hours-to-minutes-600x-faster-svm-647f904c31ae)
- [Improve the Performance of XGBoost and LightGBM Inference](https://medium.com/intel-analytics-software/improving-the-performance-of-xgboost-and-lightgbm-inference-3b542c03447e)
- [Accelerate Kaggle Challenges Using Intel AI Analytics Toolkit](https://medium.com/intel-analytics-software/accelerate-kaggle-challenges-using-intel-ai-analytics-toolkit-beb148f66d5a)
- [Accelerate Your scikit-learn Applications](https://medium.com/intel-analytics-software/improving-the-performance-of-xgboost-and-lightgbm-inference-3b542c03447e)
- [Accelerate Linear Models for Machine Learning](https://medium.com/intel-analytics-software/accelerating-linear-models-for-machine-learning-5a75ff50a0fe)
- [Accelerate K-Means Clustering](https://medium.com/intel-analytics-software/accelerate-k-means-clustering-6385088788a1)

## 🔗 Important Links
- [Notebook examples](https://github.com/intel/scikit-learn-intelex/tree/master/examples/notebooks)
- [Documentation](https://intel.github.io/scikit-learn-intelex/)
- [Supported algorithms and parameters](https://intel.github.io/scikit-learn-intelex/algorithms.html)
- [Machine Learning Benchmarks](https://github.com/IntelPython/scikit-learn_bench)

## 💬 Support

Report issues, ask questions, and provide suggestions using:

- [GitHub Issues](https://github.com/intel/scikit-learn-intelex/issues)
- [GitHub Discussions](https://github.com/intel/scikit-learn-intelex/discussions)
- [Forum](https://community.intel.com/t5/Intel-Distribution-for-Python/bd-p/distribution-python)

You may reach out to project maintainers privately at onedal.maintainers@intel.com

## daalpy and oneDAL

The acceleration is achieved through the use of the Intel(R) oneAPI Data Analytics Library (oneDAL). Learn more:
- [About Intel(R) oneAPI Data Analytics Library](https://github.com/oneapi-src/oneDAL)
- [About daal4py](https://github.com/intel/scikit-learn-intelex/tree/master/daal4py)

---
⚠️Intel(R) Extension for Scikit-learn contains scikit-learn patching functionality that was originally available in [**daal4py**](https://github.com/intel/scikit-learn-intelex/tree/master/daal4py) package. All future updates for the patches will be available only in Intel(R) Extension for Scikit-learn. We recommend you to use scikit-learn-intelex package instead of daal4py.
You can learn more about daal4py in [daal4py documentation](https://intelpython.github.io/daal4py).

---