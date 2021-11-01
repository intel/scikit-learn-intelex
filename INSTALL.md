# Installation <!-- omit in toc -->

Intel(R) Extension for Scikit-learn is available at the [Python Package Index](https://pypi.org/project/scikit-learn-intelex/),
on Anaconda Cloud in [Conda-Forge channel](https://anaconda.org/conda-forge/scikit-learn-intelex) and in [Intel channel](https://anaconda.org/intel/scikit-learn-intelex). You can also build the extension from sources.

The extension is also available as a part of [Intel® oneAPI AI Analytics Toolkit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html) (AI Kit). If you already have AI Kit installed, you do not need to separately install the extension.

## Installation scenarios <!-- omit in toc -->

- [Install via pip or conda](#install-via-pip-or-conda)
  - [Install from PyPI channel (recommended by default)](#install-from-pypi-channel-recommended-by-default)
  - [Install from Anaconda Cloud](#install-from-anaconda-cloud)
    - [Install via Anaconda Cloud from Conda-Forge channel](#install-via-anaconda-cloud-from-conda-forge-channel)
    - [Install via Anaconda Cloud from Intel channel](#install-via-anaconda-cloud-from-intel-channel)
- [Build from sources](#build-from-sources)
  - [Prerequisites](#prerequisites)
  - [Configure the build with environment variables](#configure-the-build-with-environment-variables)
  - [Build Intel(R) Extension for Scikit-learn](#build-intelr-extension-for-scikit-learn)
- [Build documentation for Intel(R) Extension for Scikit-learn](#build-documentation-for-intelr-extension-for-scikit-learn)
  - [Prerequisites for creating documentation](#prerequisites-for-creating-documentation)
  - [Build documentation](#build-documentation)

## Install via pip or conda

### Install from PyPI channel (recommended by default)

```bash
pip install scikit-learn-intelex
```

#### 📦 Supported configurations for PyPI <!-- omit in toc -->

| OS / Python version | **Python 3.6** | **Python 3.7** | **Python 3.8** | **Python 3.9** |
| :------------------ | :------------: | :------------: | :------------: | :------------: |
| **Linux**           |   [CPU, GPU]   |   [CPU, GPU]   |   [CPU, GPU]   |   [CPU, GPU]   |
| **Windows**         |   [CPU, GPU]   |   [CPU, GPU]   |   [CPU, GPU]   |   [CPU, GPU]   |
| **OsX**             |     [CPU]      |     [CPU]      |     [CPU]      |     [CPU]      |


### Install from Anaconda Cloud

#### Install via Anaconda Cloud from Conda-Forge channel

```bash
conda install scikit-learn-intelex -c conda-forge
```

##### 📦 Supported configurations for  Anaconda Cloud from Conda-Forge channel <!-- omit in toc -->

| OS / Python version     | **Python 3.6** | **Python 3.7** | **Python 3.8**| **Python 3.9**|
| :-----------------------| :------------: | :------------: | :------------:| :------------:|
|    **Linux**            |   [CPU]        |   [CPU]        |     [CPU]     |     [CPU]     |
|    **Windows**          |   [CPU]        |   [CPU]        |     [CPU]     |     [CPU]     |
|    **OsX**              |   [CPU]        |   [CPU]        |     [CPU]     |     [CPU]     |

#### Install via Anaconda Cloud from Intel channel

We recommend this installation for the users of Intel® Distribution for Python.

```bash
conda install scikit-learn-intelex -c intel
```

##### 📦 Supported configurations for Anaconda Cloud from Intel channel <!-- omit in toc -->

| OS / Python version | **Python 3.6** | **Python 3.7** | **Python 3.8** | **Python 3.9** |
| :------------------ | :------------: | :------------: | :------------: | :------------: |
| **Linux**           |   [CPU, GPU]   |   [CPU, GPU]   |   [CPU, GPU]   |   [CPU, GPU]   |
| **Windows**         |   [CPU, GPU]   |   [CPU, GPU]   |   [CPU, GPU]   |   [CPU, GPU]   |
| **OsX**             |     [CPU]      |     [CPU]      |     [CPU]      |     [CPU]      |


## Build from sources
Intel(R) Extension for Scikit-learn is easily built from sources with the majority of the necessary prerequisites available on conda or pip. The instructions below detail how to gather the prerequisites and build and install the completed package. The package can be built for all three major platforms (Windows, Linux, macOS).

### Prerequisites
* Python version >= 3.6

### Configure the build with environment variables
* SKLEARNEX_VERSION: sets package version

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
