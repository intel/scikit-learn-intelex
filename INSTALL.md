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


# Installation <!-- omit in toc -->

Intel(R) Extension for Scikit-learn is available at the [Python Package Index](https://pypi.org/project/scikit-learn-intelex/),
on Anaconda Cloud in [Conda-Forge channel](https://anaconda.org/conda-forge/scikit-learn-intelex) and in [Intel channel](https://anaconda.org/intel/scikit-learn-intelex). You can also build the extension from sources.

The extension is also available as a part of [Intel® AI Analytics Toolkit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html) (AI Kit). If you already have AI Kit installed, you do not need to separately install the extension.

## Installation scenarios <!-- omit in toc -->

- [Install via pip or conda](#install-via-pip-or-conda)
  - [Install from PyPI channel (recommended by default)](#install-from-pypi-channel-recommended-by-default)
  - [Install from Anaconda Cloud](#install-from-anaconda-cloud)
    - [Install via Anaconda Cloud from Conda-Forge channel](#install-via-anaconda-cloud-from-conda-forge-channel)
    - [Install via Anaconda Cloud from Intel channel](#install-via-anaconda-cloud-from-intel-channel)
    - [Install via Anaconda Cloud from Main channel](#install-via-anaconda-cloud-from-main-channel)
- [Build from sources](#build-from-sources)
  - [Prerequisites](#prerequisites)
  - [Configure the build with environment variables](#configure-the-build-with-environment-variables)
  - [Build Intel(R) Extension for Scikit-learn](#build-intelr-extension-for-scikit-learn)
- [Build documentation for Intel(R) Extension for Scikit-learn](#build-documentation-for-intelr-extension-for-scikit-learn)
  - [Prerequisites for creating documentation](#prerequisites-for-creating-documentation)
  - [Build documentation](#build-documentation)

Next steps after installation:

- [Learn what patching is and how to patch scikit-learn](https://intel.github.io/scikit-learn-intelex/what-is-patching.html)
- [Start using scikit-learn-intelex](https://intel.github.io/scikit-learn-intelex/quick-start.html)

## Install via pip or conda

### Install from PyPI channel (recommended by default)

1. **[Optional step] [Recommended]** To prevent version conflicts, create and activate a new environment:

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

2. Install `scikit-learn-intelex`:

   ```bash
   pip install scikit-learn-intelex
   ```

#### 📦 Supported configurations for PyPI <!-- omit in toc -->

| OS / Python version | **Python 3.6** | **Python 3.7** | **Python 3.8** | **Python 3.9** | **Python 3.10** | **Python 3.11** |
| :------------------ | :------------: | :------------: | :------------: | :------------: |  :------------: |  :------------: |
| **Linux**           |   [CPU, GPU]   |   [CPU, GPU]   |   [CPU, GPU]   |   [CPU, GPU]   |   [CPU, GPU]    |   [CPU, GPU]    | 
| **Windows**         |   [CPU, GPU]   |   [CPU, GPU]   |   [CPU, GPU]   |   [CPU, GPU]   |   [CPU, GPU]    |   [CPU, GPU]    |
| **OsX**             |     [CPU]      |     [CPU]      |     [CPU]      |     [CPU]      |     [CPU]       |     [CPU]       |


### Install from Anaconda Cloud

To prevent version conflicts, we recommend installing `scikit-learn-intelex` into a new conda environment.

#### Install via Anaconda Cloud from Conda-Forge channel

- Install into a newly created environment (recommended):

  ```bash
  conda create -n env -c conda-forge python=3.10 scikit-learn-intelex
  ```

..note:  If you do not specify the version of Python, the latest one is downloaded. 

- Install into your current environment:

  ```bash
  conda install scikit-learn-intelex -c conda-forge
  ```

##### 📦 Supported configurations for Anaconda Cloud from Conda-Forge channel <!-- omit in toc -->

| OS / Python version     | **Python 3.6** | **Python 3.7** | **Python 3.8**| **Python 3.9**| **Python 3.10**| **Python 3.11**|
| :-----------------------| :------------: | :------------: | :------------:| :------------:| :------------: |:------------:  |
|    **Linux**            |   [CPU]        |   [CPU]        |     [CPU]     |     [CPU]     |     [CPU]      |     [CPU]      |
|    **Windows**          |   [CPU]        |   [CPU]        |     [CPU]     |     [CPU]     |     [CPU]      |     [CPU]      |
|    **MacOS**            |   [CPU]        |   [CPU]        |     [CPU]     |     [CPU]     |     [CPU]      |     [CPU]      |

#### Install via Anaconda Cloud from Intel channel

We recommend this installation for the users of Intel® Distribution for Python.

- Install into a newly created environment (recommended):

  ```bash
  conda create -n env -c intel python scikit-learn-intelex
  ```

- Install into your current environment:

  ```bash
  conda install scikit-learn-intelex -c intel
  ```

##### 📦 Supported configurations for Anaconda Cloud from Intel channel <!-- omit in toc -->

| OS / Python version | **Python 3.6** | **Python 3.7** | **Python 3.8** | **Python 3.9** | **Python 3.10** |
| :------------------ | :------------: | :------------: | :------------: | :------------: | :------------:  |
| **Linux**           |   [CPU, GPU]   |   [CPU, GPU]   |   [CPU, GPU]   |   [CPU, GPU]   |   [CPU, GPU]    |
| **Windows**         |   [CPU, GPU]   |   [CPU, GPU]   |   [CPU, GPU]   |   [CPU, GPU]   |   [CPU, GPU]    |
| **MacOS**           |     [CPU]      |     [CPU]      |     [CPU]      |     [CPU]      |     [CPU]       |


#### Install via Anaconda Cloud from Main channel

- Install into a newly created environment (recommended):

  ```bash
  conda create -n env python=3.10 scikit-learn-intelex
  ```

  If you do not specify the version of Python (`python=3.10` in the example above), then latest Python is downloaded by default.

- Install into your current environment:

  ```bash
  conda install scikit-learn-intelex
  ```

##### 📦 Supported configurations for Anaconda Cloud from Main channel <!-- omit in toc -->

| OS / Python version     | **Python 3.6** | **Python 3.7** | **Python 3.8** | **Python 3.9** | **Python 3.10** | **Python 3.11** |
| :-----------------------| :------------: | :------------: | :------------: | :------------: | :------------:  | :------------:  |
| **Linux**               |   [CPU, GPU]   |   [CPU, GPU]   |   [CPU, GPU]   |   [CPU, GPU]   |   [CPU, GPU]    |   [CPU, GPU]    |
| **Windows**             |   [CPU, GPU]   |   [CPU, GPU]   |   [CPU, GPU]   |   [CPU, GPU]   |   [CPU, GPU]    |   [CPU, GPU]    |
| **MacOS**               |     [CPU]      |     [CPU]      |     [CPU]      |     [CPU]      |     [CPU]       |     [CPU]       |



## Build from sources
Intel(R) Extension for Scikit-learn is easily built from sources with the majority of the necessary prerequisites available on conda or pip. The instructions below detail how to gather the prerequisites and build and install the completed package. The package can be built for all three major platforms (Windows, Linux, macOS).

### Prerequisites
* Python version >= 3.6, <= 3.11
* daal4py >= 2021.4

**NOTE:** You can [build daal4py from sources](https://github.com/intel/scikit-learn-intelex/blob/master/daal4py/INSTALL.md) or get it from [distribution channels](https://intelpython.github.io/daal4py/#getting-daal4py).

### Configure the build with environment variables
* SKLEARNEX_VERSION: sets package version
* DALROOT: sets the oneAPI Data Analytics Library path

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

⚠️ Keys `--single-version-externally-managed` and `--no-deps` are required so that daal4py is not downloaded after installation of Intel(R) Extension for Scikit-learn

⚠️ The `develop` mode will not install the package but it will create a `.egg-link` in the deployment directory
back to the project source code directory. That way you can edit the source code and see the changes
without having to reinstall package every time you make a small change.

⚠️ `--single-version-externally-managed` is an option used for Python packages instructing the setuptools module
to create a Python package that can be easily managed by the package manager on the host.



## Build documentation for Intel(R) Extension for Scikit-learn
### Prerequisites for creating documentation

- [requirements-doc.txt](requirements-doc.txt)
- [pandoc](https://pandoc.org/installing.html)

### Build documentation

Run:

```
cd doc
./build-doc.sh
```

The documentation will be in ```doc/_build/html```.
