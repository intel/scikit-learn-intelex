.. ******************************************************************************
.. * Copyright 2020-2021 Intel Corporation
.. *
.. * Licensed under the Apache License, Version 2.0 (the "License");
.. * you may not use this file except in compliance with the License.
.. * You may obtain a copy of the License at
.. *
.. *     http://www.apache.org/licenses/LICENSE-2.0
.. *
.. * Unless required by applicable law or agreed to in writing, software
.. * distributed under the License is distributed on an "AS IS" BASIS,
.. * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. * See the License for the specific language governing permissions and
.. * limitations under the License.
.. *******************************************************************************/

.. _get_started:

############################################
Getting Intel(R) Extension for Scikit-learn*
############################################

Installation from distribution channels
---------------------------------------

Intel(R) Extension for Scikit-learn* is available at the `Python Package Index <https://pypi.org/project/scikit-learn-intelex/>`_,
and in `Intel channel <https://anaconda.org/intel/scikit-learn-intelex>`_.
Sources and build instructions are available in
`Intel(R) Extension for Scikit-learn* repository <https://github.com/intel/scikit-learn-intelex>`_.

- Install from **PyPI** (Recommended)::

     pip install scikit-learn-intelex

- Install from Anaconda Cloud: Conda-Forge channel::

     сonda install scikit-learn-intelex -c conda-forge

- Install from Anaconda Cloud: Intel channel::

    conda install scikit-learn-intelex -c intel

Supported configurations
------------------------

**PyPi channel**

.. list-table::
   :widths: 25 8 8 8 8
   :header-rows: 1
   :align: left

   * - OS / Python version
     - Python 3.6
     - Python 3.7
     - Python 3.8
     - Python 3.9
   * - Linux
     - ✔️
     - ✔️
     - ✔️
     - ❌
   * - Windows
     - ✔️
     - ✔️
     - ✔️
     - ❌
   * - OsX
     - ✔️
     - ✔️
     - ✔️
     - ❌

.. note::
    It supports Intel CPU and GPU except on OsX.

**Anaconda Cloud: Conda-Forge channel**

.. list-table::
   :widths: 25 8 8 8 8
   :header-rows: 1
   :align: left

   * - OS / Python version
     - Python 3.6
     - Python 3.7
     - Python 3.8
     - Python 3.9
   * - Linux
     - ✔️
     - ✔️
     - ✔️
     - ✔️
   * - Windows
     - ✔️
     - ✔️
     - ✔️
     - ✔️
   * - OsX
     - ✔️
     - ✔️
     - ✔️
     - ✔️

.. note::
    It supports only Intel CPU.
    Recommended for conda users by default.

**Anaconda Cloud: Intel channel**

.. list-table::
   :widths: 25 8 8 8 8
   :header-rows: 1
   :align: left

   * - OS / Python version
     - Python 3.6
     - Python 3.7
     - Python 3.8
     - Python 3.9
   * - Linux
     - ✔️
     - ✔️
     - ✔️
     - ❌
   * - Windows
     - ✔️
     - ✔️
     - ✔️
     - ❌
   * - OsX
     - ✔️
     - ✔️
     - ✔️
     - ❌

.. note::
    It supports Intel CPU and GPU except on OsX.
    Recommended for conda users who use other components from Intel(R) Distribution for Python.

Building from Sources
---------------------

**Prerequisites**::

    Python version >= 3.6

**Configuring the build with environment variables**::

    SKLEARNEX_VERSION: sets package version

**Building Intel(R) Extension for Scikit-learn**

To install the package::

    cd <checkout-dir>
    python setup_sklearnex.py install

To install the package in the development mode::

    cd <checkout-dir>
    python setup.py develop

To install scikit-learn-intelex without the download dependency of daal4py::

    cd <checkout-dir>
    python setup_sklearnex.py install --single-version-externally-managed --record=record.txt

To install scikit-learn-intelex without the download dependency of daal4py in the development mode::

    cd <checkout-dir>
    python setup_sklearnex.py develop --no-deps

.. note::
    The ``develop`` mode will not install the package but it will create a ``.egg-link`` in the deployment directory
    back to the project source code directory. That way you can edit the source code and see the changes
    without having to reinstall the package every time you make a small change.

⚠️ Keys ``--single-version-externally-managed`` and ``--no-deps`` are required so that daal4py is not downloaded after installation of Intel(R) Extension for Scikit-learn

.. note::
    ``--single-version-externally-managed`` is an option used for Python packages instructing the setuptools module
    to create a Python package that can be easily managed by the package manager on the host

**Building documentation for Intel(R) Extension for Scikit-learn**

Prerequisites for creating documentation

- sphinx
- sphinx_rtd_theme

Building documentation

1. ```cd doc && make html```
2. The documentation will be in ```doc/_build/html```
