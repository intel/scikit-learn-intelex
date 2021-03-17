# Building from sources
daal4py is easily built from sources with the majority of the necessary prerequisites available on conda. The instructions below detail how to gather the prerequisites, set your build environment, and finally build and install the completed package. daal4py can be built for all three major platforms (Windows, Linux, macOS). Multi-node (distributed) and streaming support can be disabled if needed.

The build-process (using setup.py) happens in 3 stages:
1. Creating C++ and cython sources from oneDAL C++ headers
2. Running cython on generated source
3. Compiling and linking

To build with oneAPI support, additional steps are required:
1. Point to DPC++ compiler by defining ``DPCPPROOT`` variable.

```bash
export DPCPPROOT=/opt/intel/oneapi/compiler/latest
```
2. Install Intel(R) oneAPI Data Analytics Library with oneAPI support:

    - From Conda channel.
    - From oneAPI packages repository (pass the path to oneDAL via ``DALROOT`` variable)

```bash
export DALROOT=/opt/intel/oneapi/daal/latest
```

## Building daal4py using conda-build
The easiest way to build daal4py is using the conda-build with the provided recipe.

### Prerequisites
* Python version >= 3.6
* conda-build version >= 3
* C++ compiler with C++11 support

For oneAPI support:
* A DPC++ compiler
* Intel(R) oneAPI Data Analytics Library version 2021.1 or later (https://github.com/oneapi-src/oneDAL)
  * You can use the pre-built conda package from Intel channel on anaconda.org

### Building daal4py
Library build command:
```bash
cd <checkout-dir>
conda build conda-recipe -c intel -c conda-forge
```
This will build the conda package and tell you where to find it (```.../daal4py*.tar.bz2```).

### Installing the built daal4py conda package
```
conda install <path-to-conda-package-as-built-above>
```
To actually use your daal4py, dependent packages need to be installed. To ensure, do

Linux and Windows:
```
conda install -c intel impi_rt daal numpy
```
OsX:
```
conda install -c intel -c conda-forge mpich daal numpy
```

## Building daal4py without conda-build
Without conda-build you have to manually setup your environment before building daal4py.

### Prerequisites
* Python version >= 3.6
* Jinja2
* Cython
* Numpy
* A C++ compiler with C++11 support
* [Intel® oneAPI Data Analytics Library (oneDAL)](https://github.com/oneapi-src/oneDAL) version 2021.1 or later
  * You can use the pre-built conda package from Intel channel on anaconda.org
* MPI
  * You can use the pre-built conda package intel or conda-forge channel on anaconda.org
  * Needed for distributed mode. You can disable support for distributed mode by setting NO_DIST to '1' or 'yes'
For oneAPI support
* A DPC++ compiler
* [Intel® oneAPI Data Analytics Library (oneDAL)](https://github.com/oneapi-src/oneDAL) version 2021.1 or later
  * You can use the [pre-built conda package](https://anaconda.org/intel/daal) from Intel channel on anaconda.org.

### Setting up a build environment
The easiest path for getting cython, oneDAL, MPI etc. is by creating a conda environment and setting environment variables:
```
conda create -n DAAL4PY python=3.7 impi-devel daal daal-include cython jinja2 numpy clang-tools -c intel -c conda-forge
conda activate DAAL4PY
export DALROOT=$CONDA_PREFIX
export MPIROOT=$CONDA_PREFIX
```

### Configuring the build with environment variables
* DAAL4PY_VERSION: sets package version
* NO_DIST: set to '1', 'yes' or alike to build without support for distributed mode
* NO_STREAM: set to '1', 'yes' or alike to build without support for streaming mode

### Notes on building for macOS
If building in High Sierra or higher, one may have to run into C++ build errors related to platform targets. Utilize ```export MACOSX_DEPLOYMENT_TARGET="10.9"``` if running into platform target issues.

### Building daal4py
Requires Intel(R) oneAPI Data Analytics Library and Intel(R) MPI Library being properly set up, meaning you have to set DALROOT and MPIROOT variables.

For install everything from build directory

```bash
cd <checkout-dir>
python setup.py install --single-version-externally-managed --record=record.txt
```

For install package in development mode

```bash
cd <checkout-dir>
python setup.py develop --no-deps
```

⚠️ Keys `--single-version-externally-managed` and `--no-deps` are required so that release packages are not downloads after installation of daal4py

## Building documentation for daal4py
### Prerequisites for creating documentation
* sphinx
* sphinx_rtd_theme

### Building documentation
1. Install daal4py into your python environment
2. ```cd doc && make html```
3. The documentation will be in ```doc/_build/html```
