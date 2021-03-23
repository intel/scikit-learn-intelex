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
python setup_sklearnex.py install --single-version-externally-managed --record=record.txt
```

To install the package in the development mode:

```bash
cd <checkout-dir>
python setup.py develop --no-deps
```

To install scikit-learn-intelex without the dependency on daal4py:

```bash
cd <checkout-dir>
python setup_sklearnex.py install
```
```bash
cd <checkout-dir>
python setup.py develop
```

⚠️ Keys `--single-version-externally-managed` and `--no-deps` are required so that daal4py is not downloaded after installation of Intel(R) Extension for Scikit-learn

## Building documentation for Intel(R) Extension for Scikit-learn
### Prerequisites for creating documentation
* sphinx
* sphinx_rtd_theme
* daal4py

### Building documentation
1. Install daal4py into your python environment
2. ```cd doc && make html```
3. The documentation will be in ```doc/_build/html```
