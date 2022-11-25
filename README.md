# Intel(R) Extension for Scikit-learn* <a href="#oneapi"> <img align="right" width="100" height="100" src="https://spec.oneapi.io/oneapi-logo-white-scaled.jpg"></a>

[Installation](INSTALL.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Documentation](https://intel.github.io/scikit-learn-intelex/)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Examples](https://github.com/intel/scikit-learn-intelex/tree/master/examples/notebooks)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Support](#-support)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[FAQ](#-faq)&nbsp;&nbsp;&nbsp;

[![Build Status](https://dev.azure.com/daal/daal4py/_apis/build/status/CI?branchName=master)](https://dev.azure.com/daal/daal4py/_build/latest?definitionId=9&branchName=master)
[![Coverity Scan Build Status](https://scan.coverity.com/projects/21716/badge.svg)](https://scan.coverity.com/projects/daal4py)
[![Join the community on GitHub Discussions](https://badgen.net/badge/join%20the%20discussion/on%20github/black?icon=github)](https://github.com/intel/scikit-learn-intelex/discussions)
[![PyPI Version](https://img.shields.io/pypi/v/scikit-learn-intelex)](https://pypi.org/project/scikit-learn-intelex/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/scikit-learn-intelex)](https://anaconda.org/conda-forge/scikit-learn-intelex)
[![python version](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue)](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue)
[![scikit-learn supported versions](https://img.shields.io/badge/sklearn-0.22%20%7C%200.23%20%7C%200.24%20%7C%201.0-blue)](https://img.shields.io/badge/sklearn-0.22%20%7C%200.23%20%7C%200.24%20%7C%201.0-blue)

With Intel(R) Extension for Scikit-learn you can accelerate your Scikit-learn applications and still have full conformance with all Scikit-Learn APIs and algorithms. This is a **free software AI accelerator** that brings over **10-100X** acceleration across a variety of applications. And you do not even need to change the existing code!

## How it works?

Intel(R) Extension for Scikit-learn offers you a way to accelerate existing scikit-learn code.
The acceleration is achieved through **patching**: replacing the stock scikit-learn algorithms with their optimized versions provided by the extension.

One of the ways to patch scikit-learn is by modifying the code. First, you import an additional Python package (`sklearnex`) and enable optimizations via `sklearnex.patch_sklearn()`. Then import scikit-learn estimators:

- **Enable Intel CPU optimizations**

    ```py
    import numpy as np
    from sklearnex import patch_sklearn
    patch_sklearn()

    from sklearn.cluster import DBSCAN

    X = np.array([[1., 2.], [2., 2.], [2., 3.],
                [8., 7.], [8., 8.], [25., 80.]], dtype=np.float32)
    clustering = DBSCAN(eps=3, min_samples=2).fit(X)
    ```

- **Enable Intel GPU optimizations**

    ```py
    import numpy as np
    import dpctl
    from sklearnex import patch_sklearn, config_context
    patch_sklearn()

    from sklearn.cluster import DBSCAN

    X = np.array([[1., 2.], [2., 2.], [2., 3.],
                [8., 7.], [8., 8.], [25., 80.]], dtype=np.float32)
    with config_context(target_offload="gpu:0"):
        clustering = DBSCAN(eps=3, min_samples=2).fit(X)
    ```

👀 Read about [other ways to patch scikit-learn](https://intel.github.io/scikit-learn-intelex/index.html#usage) and [other methods for offloading to GPU devices](https://intel.github.io/scikit-learn-intelex/oneapi-gpu.html).
Check out available [notebooks](https://github.com/intel/scikit-learn-intelex/tree/master/examples/notebooks) for more examples.

This software acceleration is achieved through the use of vector instructions, IA hardware-specific memory optimizations, threading, and optimizations for all upcoming Intel platforms at launch time.

## Supported Algorithms

❗ The patching only affects [selected algorithms and their parameters](https://intel.github.io/scikit-learn-intelex/algorithms.html).

You may still use algorithms and parameters not supported by Intel(R) Extension for Scikit-learn in your code. You will not get an error if you do this. When you use algorithms or parameters not supported by the extension, the package fallbacks into original stock version of scikit-learn.

## 🚀 Acceleration

![](https://raw.githubusercontent.com/intel/scikit-learn-intelex/master/doc/sources/_static/scikit-learn-acceleration-2021.2.3.PNG)
Configurations:
- HW: c5.24xlarge AWS EC2 Instance using an Intel Xeon Platinum 8275CL with 2 sockets and 24 cores per socket
- SW: scikit-learn version 0.24.2, scikit-learn-intelex version 2021.2.3, Python 3.8

[Benchmarks code](https://github.com/IntelPython/scikit-learn_bench)

## 🛠 Installation

[System Requirements](https://intel.github.io/scikit-learn-intelex/system-requirements.html)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp; [Install via pip or conda](https://github.com/intel/scikit-learn-intelex/blob/master/INSTALL.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Build from sources](INSTALL.md#build-from-sources)

Intel(R) Extension for Scikit-learn is available at the [Python Package Index](https://pypi.org/project/scikit-learn-intelex/),
on Anaconda Cloud in [Conda-Forge channel](https://anaconda.org/conda-forge/scikit-learn-intelex) and in [Intel channel](https://anaconda.org/intel/scikit-learn-intelex). You can also build the extension from [sources](INSTALL.md#build-from-sources).

The extension is also available as a part of [Intel® AI Analytics Toolkit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html) (AI Kit). If you already have AI Kit installed, you do not need to install the extension.

Installation via `pip` package manager is recommended by default:

```bash
pip install scikit-learn-intelex
```

## 🔗 Important Links
- [Notebook examples](https://github.com/intel/scikit-learn-intelex/tree/master/examples/notebooks)
- [Documentation](https://intel.github.io/scikit-learn-intelex/)
- [Supported algorithms and parameters](https://intel.github.io/scikit-learn-intelex/algorithms.html)
- [Machine Learning Benchmarks](https://github.com/IntelPython/scikit-learn_bench)

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

## 💬 Support

Report issues, ask questions, and provide suggestions using:

- [GitHub Issues](https://github.com/intel/scikit-learn-intelex/issues)
- [GitHub Discussions](https://github.com/intel/scikit-learn-intelex/discussions)
- [Forum](https://community.intel.com/t5/Intel-Distribution-for-Python/bd-p/distribution-python)

You may reach out to project maintainers privately at onedal.maintainers@intel.com

## oneAPI

Intel(R) Extension for Scikit-learn is part of [oneAPI](https://oneapi.io) and [Intel® AI Analytics Toolkit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html) (AI Kit).

## daalpy and oneDAL

The acceleration is achieved through the use of the Intel(R) oneAPI Data Analytics Library (oneDAL). Learn more:
- [About Intel(R) oneAPI Data Analytics Library](https://github.com/oneapi-src/oneDAL)
- [About daal4py](https://github.com/intel/scikit-learn-intelex/tree/master/daal4py)

---
⚠️Intel(R) Extension for Scikit-learn contains scikit-learn patching functionality that was originally available in [**daal4py**](https://github.com/intel/scikit-learn-intelex/tree/master/daal4py) package. All future updates for the patches will be available only in Intel(R) Extension for Scikit-learn. We recommend you to use scikit-learn-intelex package instead of daal4py.
You can learn more about daal4py in [daal4py documentation](https://intelpython.github.io/daal4py).

---

