# Building from sources
Intel(R) Extension for Scikit-learn is easily built from sources with the majority of the necessary prerequisites available on conda or pip. The instructions below detail how to gather the prerequisites, and finally build and install the completed package. The package can be built for all three major platforms (Windows, Linux, macOS).

### Prerequisites
* Python version >= 3.6

### Configuring the build with environment variables
* SKLEARNEX_VERSION: sets package version

### Building Intel(R) Extension for Scikit-learn
To install the package:

```bash
cd <checkout-dir>
python setup_sklearnex.py install
```

To install the package in the development mode:

```bash
cd <checkout-dir>
python setup_sklearnex.py develop
```

To install scikit-learn-intelex without the dependency on daal4py:

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

## Building documentation for Intel(R) Extension for Scikit-learn
### Prerequisites for creating documentation
* sphinx
* sphinx_rtd_theme

### Building documentation
1. ```cd doc && make html```
2. The documentation will be in ```doc/_build/html```
