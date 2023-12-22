<div align="center">


# Intel(R) Extension for Scikit-learn* 

<h3> Speed up your scikit-learn applications for Intel(R) CPUs and GPUs across single- and multi-node configurations

[Releases](https://github.com/intel/scikit-learn-intelex/releases)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Documentation](https://intel.github.io/scikit-learn-intelex/)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Examples](https://github.com/intel/scikit-learn-intelex/tree/master/examples/notebooks)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Support]()&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;[License](https://github.com/intel/scikit-learn-intelex/blob/master/LICENSE)&nbsp;&nbsp;&nbsp;


[![Build Status](https://dev.azure.com/daal/daal4py/_apis/build/status/CI?branchName=main)](https://dev.azure.com/daal/daal4py/_build/latest?definitionId=9&branchName=main)
[![Coverity Scan Build Status](https://scan.coverity.com/projects/21716/badge.svg)](https://scan.coverity.com/projects/daal4py)
[![Join the community on GitHub Discussions](https://badgen.net/badge/join%20the%20discussion/on%20github/black?icon=github)](https://github.com/intel/scikit-learn-intelex/discussions)
[![PyPI Version](https://img.shields.io/pypi/v/scikit-learn-intelex)](https://pypi.org/project/scikit-learn-intelex/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/scikit-learn-intelex)](https://anaconda.org/conda-forge/scikit-learn-intelex)
[![python version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)
[![scikit-learn supported versions](https://img.shields.io/badge/sklearn-1.0%20%7C%201.1%20%7C%201.2%20%7C%201.3-blue)](https://img.shields.io/badge/sklearn-01.0%20%7C%201.1%20%7C%201.2%20%7C%201.3-blue)

---
</h3>

<div align="left">

## Overview

Intel(R) Extension for Scikit-learn is a **free software AI accelerator** designed to deliver over **10-100X** acceleration to your existing scikit-learn code.
The software acceleration is achieved with vector instructions, AI hardware-specific memory optimizations, threading, and optimizations for all upcoming Intel(R) platforms at launch time.


With Intel(R) Extension for Scikit-learn, you can:

* Speed up training and inference by up to 100x with the equivalent mathematical accuracy
* Benefit from performance improvements across different Intel(R) hardware configurations
* Integrate the extension into your existing Scikit-learn applications without code modifications
* Continue to use the open-source scikit-learn API
* Enable and disable the extension with a couple of lines of code or at the command line

Intel(R) Extension for Scikit-learn is also a part of [Intel(R) AI Tools](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html). 

## Acceleration

![](https://raw.githubusercontent.com/intel/scikit-learn-intelex/master/doc/sources/_static/scikit-learn-acceleration.PNG)

[Benchmarks code](https://github.com/IntelPython/scikit-learn_bench)

## Intel(R) Optimizations

- **Enable Intel(R) CPU optimizations**

    ```py
    import numpy as np
    from sklearnex import patch_sklearn
    patch_sklearn()

    from sklearn.cluster import DBSCAN

    X = np.array([[1., 2.], [2., 2.], [2., 3.],
                [8., 7.], [8., 8.], [25., 80.]], dtype=np.float32)
    clustering = DBSCAN(eps=3, min_samples=2).fit(X)
    ```

- **Enable Intel(R) GPU optimizations**

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
:eyes: Check out available [notebooks](https://github.com/intel/scikit-learn-intelex/tree/master/examples/notebooks) for more examples.

## Installation

To install Intel(R) Extension for Scikit-learn, run:

```
pip install scikit-learn-intelex
```

See all installation instructions in the [Installation Guide](https://intel.github.io/scikit-learn-intelex/latest/installation.html).

## Integration

The software acceleration is achieved through patching. It means, replacing the stock scikit-learn algorithms with their optimized versions provided by the extension.

The patching only affects [supported algorithms and their parameters](https://intel.github.io/scikit-learn-intelex/latest/algorithms.html).
You can still use not supported ones in your code, the package simply fallbacks into the stock version of scikit-learn. 

> **_TIP:_** Enable [verbose mode](https://intel.github.io/scikit-learn-intelex/latest/verbose.html) to see which implementation of the algorithm is currently used.

To patch scikit-learn, you can:
* Use the following command-line flag:
  ```
  python -m sklearnex my_application.py
  ```
* Add the following lines to the script:
  ```
  from sklearnex import patch_sklearn
  patch_sklearn()
  ```

:eyes: Read about [other ways to patch scikit-learn](https://intel.github.io/scikit-learn-intelex/index.html#usage). 

## Documentation

* [Quick Start](https://intel.github.io/scikit-learn-intelex/latest/quick-start.html)
* [Documentation and Tutorials](https://intel.github.io/scikit-learn-intelex/latest/index.html)
* [Release Notes](https://github.com/intel/scikit-learn-intelex/releases)
* [Medium Blogs](https://intel.github.io/scikit-learn-intelex/latest/blogs.html)
* [Code of Conduct](https://github.com/intel/scikit-learn-intelex/blob/master/CODE_OF_CONDUCT.md)

### daal4py and oneDAL

The acceleration is achieved through the use of the Intel(R) oneAPI Data Analytics Library (oneDAL). Learn more:
- [About Intel(R) oneAPI Data Analytics Library](https://github.com/oneapi-src/oneDAL)
- [About daal4py](https://github.com/intel/scikit-learn-intelex/tree/main/daal4py)

## Samples & Examples

* [Examples](https://github.com/intel/scikit-learn-intelex/tree/master/examples/notebooks)
* [Samples](https://intel.github.io/scikit-learn-intelex/latest/samples.html)
* [Kaggle Kernels](https://intel.github.io/scikit-learn-intelex/latest/kaggle.html)


## How to Contribute

We welcome community contributions, check our [Contributing Guidelines](https://github.com/intel/scikit-learn-intelex/blob/master/CONTRIBUTING.md) to learn more.

------------------------------------------------------------------------
\* The Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries. Other names and brands may be claimed as the property of others. 

