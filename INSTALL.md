<!--
******************************************************************************
* Copyright 2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/-->

  
# Installation  <!-- omit in toc -->

To install Intel(R) Extension for Scikit-learn*, use one of the following scenarios:

- [Before You Begin](#before-you-begin)
- [Install via PIP](#install-via-pip)
  - [Install from PyPI Channel (recommended by default)](#install-from-pypi-channel-recommended-by-default)
- [Install from Anaconda Cloud](#install-from-anaconda-cloud)
  - [Install via Anaconda Cloud from Conda-Forge Channel](#install-via-anaconda-cloud-from-conda-forge-channel)
  - [Install via Anaconda Cloud from Intel Channel](#install-via-anaconda-cloud-from-intel-channel)
- [Build from Sources](#build-from-sources)
  - [Prerequisites](#prerequisites)
  - [Configure the Build with Environment Variables](#configure-the-build-with-environment-variables)
  - [Build Intel(R) Extension for Scikit-learn](#build-intelr-extension-for-scikit-learn)
- [Build from Sources with `conda-build`](#build-from-sources-with-conda-build)
  - [Prerequisites for `conda-build`](#prerequisites-for-conda-build)
  - [Build Intel(R) Extension for Scikit-learn with `conda-build`](#build-intelr-extension-for-scikit-learn-with-conda-build)
- [Next Steps](#next-steps)

> **_NOTE:_** Intel(R) Extension for Scikit-learn* is also available as a part of [Intel® AI Tools](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html). If you already have it installed, you do not need to separately install the extension.


## Before You Begin

Check [System](https://intel.github.io/scikit-learn-intelex/latest/system-requirements.html) and [Memory](https://intel.github.io/scikit-learn-intelex/latest/memory-requirements.html) Requirements.

## Supported Configurations

| OS / Python version | **Python 3.9** | **Python 3.10** | **Python 3.11** | **Python 3.12** |
| :------------------ | :------------: |  :------------: |  :------------: |  :------------: |
| **Linux**           |   [CPU, GPU]   |   [CPU, GPU]    |   [CPU, GPU]    |   [CPU, GPU]    |
| **Windows**         |   [CPU, GPU]   |   [CPU, GPU]    |   [CPU, GPU]    |   [CPU, GPU]    |

Applicable for:

* PyPI
* Anaconda Cloud from Conda-Forge Channel
* Anaconda Cloud from Intel Channel



## Install via PIP

To prevent version conflicts, create and activate a new environment:

   - On Linux:

     ```bash
     python -m venv env
     source env/bin/activate
     ```

   - On Windows:

     ```bash
     python -m venv env
     .\env\Scripts\activate
     ```

### Install from PyPI Channel (recommended by default)

Install `scikit-learn-intelex`:

   ```bash
   pip install scikit-learn-intelex
   ```

## Install from Anaconda Cloud

To prevent version conflicts, we recommend to create and activate a new environment. 

### Install via Anaconda Cloud from Conda-Forge Channel

- Install into a newly created environment (recommended):

  ```bash
  conda config --add channels conda-forge
  conda config --set channel_priority strict
  conda create -n env python=3.10 scikit-learn-intelex
  ```

> **_NOTE:_** If you do not specify the Python version, the latest one is downloaded. 

- Install into your current environment:

  ```bash
  conda config --add channels conda-forge
  conda config --set channel_priority strict
  conda install scikit-learn-intelex
  ```

### Install via Anaconda Cloud from Intel Channel

We recommend this installation for the users of Intel® Distribution for Python.

- Install into a newly created environment (recommended):

  ```bash
  conda config --add channels https://software.repos.intel.com/python/conda/
  conda config --set channel_priority strict
  conda create -n env python=3.10 scikit-learn-intelex
  ```

> **_NOTE:_** If you do not specify the Python version, the latest one is downloaded. 

- Install into your current environment:

  ```bash
  conda config --add channels https://software.repos.intel.com/python/conda/
  conda config --set channel_priority strict
  conda install scikit-learn-intelex
  ```

> **_NOTE:_** If you do not specify the version of Python, the latest one is downloaded. 

- Install into your current environment:

  ```bash
  conda install scikit-learn-intelex
  ```

## Build from Sources
Intel(R) Extension for Scikit-learn* is easily built from the sources with the majority of the necessary prerequisites available with conda or pip. 

The package is available for Windows* OS, Linux* OS, and macOS*.

Multi-node (distributed) and streaming support can be disabled if needed.

The build-process (using setup.py) happens in 4 stages:
1. Creating C++ and Cython sources from oneDAL C++ headers
2. Building oneDAL Python interfaces via cmake and pybind11
3. Running Cython on generated sources
4. Compiling and linking them

### Prerequisites
* Python version >= 3.9, <= 3.12
* Jinja2
* Cython
* Numpy
* cmake and pybind11
* A C++ compiler with C++11 support
* Clang-Format
* [Intel® oneAPI Data Analytics Library (oneDAL)](https://github.com/oneapi-src/oneDAL) version 2021.1 or later
  * You can use the pre-built `dal-devel` conda package from conda-forge channel
* MPI (optional, needed for distributed mode)
  * You can use the pre-built `impi_rt` and `impi-devel` conda packages from conda-forge channel
* A DPC++ compiler (optional, needed for DPC++ interfaces)

### Configure the Build with Environment Variables
* ``SKLEARNEX_VERSION``: sets the package version
* ``DALROOT``: sets the oneAPI Data Analytics Library path
* ``MPIROOT``: sets the path to the MPI library that will be used for distributed mode support. Not used when using `NO_DIST=1`
* ``NO_DIST``: set to '1', 'yes' or alike to build without support for distributed mode
* ``NO_STREAM``: set to '1', 'yes' or alike to build without support for streaming mode
* ``NO_DPC``: set to '1', 'yes' or alike to build without support of oneDAL DPC++ interfaces
* ``OFF_ONEDAL_IFACE``: set to '1' to build without the support of oneDAL interfaces

**Note:** in order to use distributed mode, `mpi4py` is also required, and needs to be built with the same MPI backend as scikit-learn-intelex.

### Build Intel(R) Extension for Scikit-learn

- To install the package:

   ```bash
   cd <checkout-dir>
   python setup.py install
   ```

- To install the package in the development mode:

   ```bash
   cd <checkout-dir>
   python setup.py develop
   ```

- To install scikit-learn-intelex without checking for dependencies:

   ```bash
   cd <checkout-dir>
   python setup.py install --single-version-externally-managed --record=record.txt
   ```
   ```bash
   cd <checkout-dir>
   python setup.py develop --no-deps
   ```

Where: 

* Keys `--single-version-externally-managed` and `--no-deps` are required to not download daal4py after the installation of Intel(R) Extension for Scikit-learn. 
* The `develop` mode does not install the package but creates a `.egg-link` in the deployment directory
back to the project source-code directory. That way, you can edit the source code and see the changes
without reinstalling the package after a small change.
* `--single-version-externally-managed` is an option for Python packages instructing the setuptools module to create a package that the host's package manager can easily manage.

- To build the python module without installing it:

   ```bash
   cd <checkout-dir>
   python setup.py build_ext --inplace --force
   python setup.py build
   ```

**Note:** the `setup.py` file will accept an optional argument `--abs-rpath` on linux (for all of `build`/`install`/`develop`/etc.) which will make it add the absolute path to oneDAL's shared objects (.so files) to the rpath of the scikit-learn-intelex extension's shared object files in order to load them automatically. This is not necessary when installing from pip or conda, but can be helpful for development purposes when using a from-source build of oneDAL that resides in a custom folder, as it won't assume that oneDAL's files will be found under default system paths. Example:

```shell
python setup.py build_ext --inplace --force --abs-rpath
python setup.py build --abs-rpath
```

**Note:** when building `scikit-learn-intelex` from source with this option, it will use the oneDAL library with which it was compiled. oneDAL has dependencies on other libraries such as TBB, which is also distributed as a python package through `pip` and as a `conda` package. By default, a conda environment will first try to load TBB from its own packages if it is installed in the environment, which might cause issues if oneDAL was compiled with a system TBB instead of a conda one. In such cases, it is advised to either uninstall TBB from pip/conda (it will be loaded from the oneDAL library which links to it), or modify the order of search paths in environment variables like `${LD_LIBRARY_PATH}`.

## Build from Sources with `conda-build`

Intel(R) Extension for Scikit-learn* is easily built from the sources using only one command and `conda-build` utility. 

### Prerequisites for `conda-build`

* any `conda` distribution (`miniforge` is recommended)
* `conda-build` and `conda-verify` installed in a conda environment
* (Windows only) Microsoft Visual Studio*
* (optional) Intel(R) oneAPI DPC++/C++ Compiler

`conda-build` config requires **2022** version of Microsoft Visual Studio* by default, you can specify another version in `conda-recipe/conda_build_config.yaml` if needed.

In order to enable DPC++ interfaces support on Windows, you need to set `DPCPPROOT` environment variable pointing to DPC++/C++ Compiler distribution.
Conda-forge distribution of DPC++ compiler is used by default on Linux, but you still can set your own distribution via `DPCPPROOT` variable.

### Build Intel(R) Extension for Scikit-learn with `conda-build`

Create and verify `scikit-learn-intelex` conda package with next command executed from root of sklearnex repo:

```bash
conda build .
```

## Next Steps

- [Learn what patching is and how to patch scikit-learn](https://intel.github.io/scikit-learn-intelex/latest/what-is-patching.html)
- [Start using scikit-learn-intelex](https://intel.github.io/scikit-learn-intelex/latest/quick-start.html)
