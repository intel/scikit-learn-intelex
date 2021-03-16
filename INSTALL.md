# Building from sources
scikit-learn-intelex is easily built from sources with the majority of the necessary prerequisites available on conda. The instructions below detail how to gather the prerequisites, set your build environment, and finally build and install the completed package. scikit-learn-intelex can be built for all three major platforms (Windows, Linux, macOS). Multi-node (distributed) and streaming support can be disabled if needed.


### Prerequisites
* Python version >= 3.6

### Setting up a build environment

### Configuring the build with environment variables
* SKLEARNEX_VERSION: sets package version

### Building scikit-learn-intelex
For install scikit-learn-intelex

```bash
cd <checkout-dir>
python setup_sklearnex.py install --single-version-externally-managed --record=record.txt
```

For install package in development mode

```bash
cd <checkout-dir>
python setup.py develop --no-deps
```

For install scikit-learn-intelex without main daal4py dependency

```bash
cd <checkout-dir>
python setup_sklearnex.py install
```
```bash
cd <checkout-dir>
python setup.py develop
```

⚠️ Keys `--single-version-externally-managed` and `--no-deps` are required so that release daal4py are not downloads after installation of scikit-learn-intelex

## Building documentation for scikit-learn-intelex
### Prerequisites for creating documentation
* sphinx
* sphinx_rtd_theme

### Building documentation
1. Install daal4py into your python environment
2. ```cd doc && make html```
3. The documentation will be in ```doc/_build/html```
