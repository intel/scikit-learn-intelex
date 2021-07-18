# Intel(R) Extension for Scikit-learn*
[![Build Status](https://dev.azure.com/daal/daal4py/_apis/build/status/CI?branchName=master)](https://dev.azure.com/daal/daal4py/_build/latest?definitionId=9&branchName=master)
[![Coverity Scan Build Status](https://scan.coverity.com/projects/21716/badge.svg)](https://scan.coverity.com/projects/daal4py)
[![Join the community on GitHub Discussions](https://badgen.net/badge/join%20the%20discussion/on%20github/black?icon=github)](https://github.com/intel/scikit-learn-intelex/discussions)
[![PyPI Version](https://img.shields.io/pypi/v/scikit-learn-intelex)](https://pypi.org/project/scikit-learn-intelex/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/scikit-learn-intelex)](https://anaconda.org/conda-forge/scikit-learn-intelex)

Intel(R) Extension for Scikit-learn is a seamless way to speed up your Scikit-learn application. The acceleration is achieved through the use of the Intel(R) oneAPI Data Analytics Library ([oneDAL](https://github.com/oneapi-src/oneDAL)). Patching scikit-learn makes it a well-suited machine learning framework for dealing with real-life problems.

⚠️Intel(R) Extension for Scikit-learn contains scikit-learn patching functionality that was originally available in [**daal4py**](https://github.com/intel/scikit-learn-intelex/tree/master/daal4py) package. All future updates for the patches will be available only in Intel(R) Extension for Scikit-learn. We recommend you to use scikit-learn-intelex package instead of daal4py.
You can learn more about daal4py in [daal4py documentation](https://intelpython.github.io/daal4py).

Running the latest scikit-learn test suite with Intel(R) Extension for Scikit-learn: [![CircleCI](https://circleci.com/gh/intel/scikit-learn-intelex.svg?style=svg)](https://circleci.com/gh/intel/scikit-learn-intelex)

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

## 🔗 Important links
- [Notebook examples](https://github.com/intel/scikit-learn-intelex/tree/master/examples/notebooks)
- [Documentation](https://intel.github.io/scikit-learn-intelex/)
- [scikit-learn API and patching](https://intel.github.io/scikit-learn-intelex/)
- [Benchmark code](https://github.com/IntelPython/scikit-learn_bench)
- [Building from Sources](https://github.com/intel/scikit-learn-intelex/blob/master/INSTALL.md)
- [About Intel(R) oneAPI Data Analytics Library](https://github.com/oneapi-src/oneDAL)
- [About Intel(R) daal4py](https://github.com/intel/scikit-learn-intelex/tree/master/daal4py)

## 💬 Support

Report issues, ask questions, and provide suggestions using:

- [GitHub Issues](https://github.com/intel/scikit-learn-intelex/issues)
- [GitHub Discussions](https://github.com/intel/scikit-learn-intelex/discussions)
- [Forum](https://community.intel.com/t5/Intel-Distribution-for-Python/bd-p/distribution-python)

You may reach out to project maintainers privately at onedal.maintainers@intel.com

# 🛠 Installation
Intel(R) Extension for Scikit-learn is available at the [Python Package Index](https://pypi.org/project/scikit-learn-intelex/),
on Anaconda Cloud in [Conda-Forge channel](https://anaconda.org/conda-forge/scikit-learn-intelex) and in [Intel channel](https://anaconda.org/intel/scikit-learn-intelex).
Intel(R) Extension for Scikit-learn is also available as a part of [Intel® oneAPI AI Analytics Toolkit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html) (AI Kit).

```bash
# PyPi (recommended by default)
pip install scikit-learn-intelex
```

```bash
# Anaconda Cloud from Conda-Forge channel (recommended for conda users by default)
conda install scikit-learn-intelex -c conda-forge
```

```bash
# Anaconda Cloud from Intel channel (recommended for Intel® Distribution for Python users)
conda install scikit-learn-intelex -c intel
```

<details><summary>[Click to expand] ℹ️ Supported configurations </summary>

#### 📦 PyPi channel

| OS / Python version     | **Python 3.6** | **Python 3.7** | **Python 3.8**| **Python 3.9**|
| :-----------------------| :------------: | :-------------:| :------------:| :------------:|
|    **Linux**            |    [CPU, GPU]  |  [CPU, GPU]    |   [CPU, GPU]  |     ❌       |
|    **Windows**          |    [CPU, GPU]  |  [CPU, GPU]    |   [CPU, GPU]  |     ❌       |
|    **OsX**              |    [CPU]       |  [CPU]         |    [CPU]      |     ❌       |

#### 📦 Anaconda Cloud: Conda-Forge channel

| OS / Python version     | **Python 3.6** | **Python 3.7** | **Python 3.8**| **Python 3.9**|
| :-----------------------| :------------: | :------------: | :------------:| :------------:|
|    **Linux**            |   [CPU]        |   [CPU]        |     [CPU]     |     [CPU]     |
|    **Windows**          |   [CPU]        |   [CPU]        |     [CPU]     |     [CPU]     |
|    **OsX**              |   [CPU]        |   [CPU]        |     [CPU]     |     [CPU]     |

#### 📦 Anaconda Cloud: Intel channel

| OS / Python version     | **Python 3.6** | **Python 3.7** | **Python 3.8**| **Python 3.9**|
| :-----------------------| :------------: | :-------------:| :------------:| :------------:|
|    **Linux**            |   [CPU, GPU]   |     [CPU, GPU]  |  [CPU, GPU]  |      ❌       |
|    **Windows**          |   [CPU, GPU]   |     [CPU, GPU]  |  [CPU, GPU]  |      ❌       |
|    **OsX**              |   [CPU]        |     [CPU]       |   [CPU]      |      ❌       |

</details>

⚠️ Note: *GPU support is an optional dependency. Required dependencies for GPU support
will not be downloaded. You need to manually install ***dpcpp_cpp_rt*** package.*

<details><summary>[Click to expand] ℹ️ How to install dpcpp_cpp_rt package </summary>

```bash
# PyPi
pip install --upgrade dpcpp_cpp_rt
```

```bash
# Anaconda Cloud
conda install dpcpp_cpp_rt -c intel
```

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

![](https://raw.githubusercontent.com/intel/scikit-learn-intelex/master/doc/sources/_static/scikit-learn-acceleration-2021.2.3.PNG)
Configurations:
- HW: c5.24xlarge AWS EC2 Instance using an Intel Xeon Platinum 8275CL with 2 sockets and 24 cores per socket
- SW: scikit-learn version 0.24.2, scikit-learn-intelex version 2021.2.3, Python 3.8

[Benchmarks code](https://github.com/IntelPython/scikit-learn_bench)

<details><summary>[Click to expand] ℹ️ Reproduce results </summary>

```bash
# With Intel® Extension for Scikit-learn enabled:
python runner.py --configs configs/blogs/skl_conda_config.json –report
```
```bash
# With the original Scikit-learn:
python runner.py --configs configs/blogs/skl_conda_config.json –report --no-intel-optimized
```
</details>

Intel(R) Extension for Scikit-learn patching affects performance of specific Scikit-learn functionality. Refer to the [list of supported algorithms and parameters](https://intel.github.io/scikit-learn-intelex/algorithms.html) for details. In cases when unsupported parameters are used, the package fallbacks into original Scikit-learn. If the patching does not cover your scenarios, [submit an issue on GitHub](https://github.com/intel/scikit-learn-intelex/issues).

⚠️ We support optimizations for the last four versions of scikit-learn. The latest release of Intel(R) Extension for Scikit-learn 2021.3.X supports scikit-learn 0.22.X,
0.23.X, 0.24.X and 1.0.X.

## 📜 Intel(R) Extension for Scikit-learn verbose

To find out which implementation of the algorithm is currently used (Intel(R) Extension for Scikit-learn or original Scikit-learn), set the environment variable:
- On Linux and Mac OS: `export SKLEARNEX_VERBOSE=INFO`
- On Windows: `set SKLEARNEX_VERBOSE=INFO`

For example, for DBSCAN you get one of these print statements depending on which implementation is used:
- `SKLEARNEX INFO: sklearn.cluster.DBSCAN.fit: running accelerated version on CPU`
- `SKLEARNEX INFO: sklearn.cluster.DBSCAN.fit: fallback to original Scikit-learn`

[Read more in the documentation](https://intel.github.io/scikit-learn-intelex/).
