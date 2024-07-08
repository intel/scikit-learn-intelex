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
- [Next Steps](#next-steps)

> **_NOTE:_** Intel(R) Extension for Scikit-learn* is also available as a part of [Intel® AI Tools](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html). If you already have it installed, you do not need to separately install the extension.


## Before You Begin

Check [System](https://intel.github.io/scikit-learn-intelex/latest/system-requirements.html) and [Memory](https://intel.github.io/scikit-learn-intelex/latest/memory-requirements.html) Requirements.

## Supported Configurations

| OS / Python version | **Python 3.8** | **Python 3.9** | **Python 3.10** | **Python 3.11** | **Python 3.12** |
| :------------------ | :------------: | :------------: |  :------------: |  :------------: |  :------------: |
| **Linux**           |   [CPU, GPU]   |   [CPU, GPU]   |   [CPU, GPU]    |   [CPU, GPU]    |   [CPU, GPU]    |
| **Windows**         |   [CPU, GPU]   |   [CPU, GPU]   |   [CPU, GPU]    |   [CPU, GPU]    |   [CPU, GPU]    |

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
  conda config --add channels intel
  conda config --set channel_priority strict
  conda create -n env python=3.10 scikit-learn-intelex
  ```

> **_NOTE:_** If you do not specify the Python version, the latest one is downloaded. 

- Install into your current environment:

  ```bash
  conda config --add channels intel
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

### Prerequisites
* Python version >= 3.8, <= 3.12
* daal4py >= 2024.3

**TIP:** [Build daal4py from sources](https://github.com/intel/scikit-learn-intelex/blob/main/daal4py/INSTALL.md) or get it from [distribution channels](https://intelpython.github.io/daal4py/#getting-daal4py).

### Configure the Build with Environment Variables
* ``SKLEARNEX_VERSION``: sets the package version
* ``DALROOT``: sets the oneAPI Data Analytics Library path

### Build Intel(R) Extension for Scikit-learn

- To install the package:

   ```bash
   cd <checkout-dir>
   python setup_sklearnex.py install
   ```

- To install the package in the development mode:

   ```bash
   cd <checkout-dir>
   python setup_sklearnex.py develop
   ```

- To install scikit-learn-intelex without downloading daal4py:

   ```bash
   cd <checkout-dir>
   python setup_sklearnex.py install --single-version-externally-managed --record=record.txt
   ```
   ```bash
   cd <checkout-dir>
   python setup_sklearnex.py develop --no-deps
   ```

Where: 

* Keys `--single-version-externally-managed` and `--no-deps` are required to not download daal4py after the installation of Intel(R) Extension for Scikit-learn. 
* The `develop` mode does not install the package but creates a `.egg-link` in the deployment directory
back to the project source-code directory. That way, you can edit the source code and see the changes
without reinstalling the package after a small change.
* `--single-version-externally-managed` is an option for Python packages instructing the setup tools module to create a package the host's package manager can easily manage.

## Next Steps

- [Learn what patching is and how to patch scikit-learn](https://intel.github.io/scikit-learn-intelex/latest/what-is-patching.html)
- [Start using scikit-learn-intelex](https://intel.github.io/scikit-learn-intelex/latest/quick-start.html)
