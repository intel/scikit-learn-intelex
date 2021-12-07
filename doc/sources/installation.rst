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

.. |intelex_repo| replace:: |intelex| repository
.. _intelex_repo: https://github.com/intel/scikit-learn-intelex

#############
Installation
#############

|intelex| is available at the `Python Package Index <https://pypi.org/project/scikit-learn-intelex/>`_,
on Anaconda Cloud in `Conda-Forge channel <https://anaconda.org/conda-forge/scikit-learn-intelex>`_ and
in `Intel channel <https://anaconda.org/intel/scikit-learn-intelex>`_. 

|intelex| is also available as a part of `Intel oneAPI AI Analytics Toolkit
<https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html#gs.3lkbv3>`_ (AI Kit).
If you already have AI Kit installed, you do not need to separately install the extension.

You can also build the extension from :ref:`sources <build_from_sources>`.

Install from distribution channels
-----------------------------------

.. rubric:: Install from PyPI (recommended by default)

#. [Optional step] [Recommended] To prevent version conflicts, create and activate a new environment:

   .. tabs::

      .. tab:: Linux

         ::
            
            python -m venv env
            source env/bin/activate

      .. tab:: Windows

         ::

            python -m venv env
            .\env\Scripts\activate

#. Install `scikit-learn-intelex`:

   ::

      pip install scikit-learn-intelex

.. rubric:: Install from Anaconda Cloud

To prevent version conflicts, we recommend installing `scikit-learn-intelex` into a new conda environment.
For each distribution channel, there are two installation options: with and without a creation of a new environment.


- Install from **Anaconda Cloud: Conda-Forge channel** (recommended by default for conda users):

  - into a newly created environment::

       conda create -n env -c conda-forge python scikit-learn-intelex

  - into your current environment::
    
       conda install scikit-learn-intelex -c conda-forge

- Install from **Anaconda Cloud: Intel channel** (recommended for the users of Intel® Distribution for Python):

  - into a newly created environment::
    
       conda create -n env -c intel python scikit-learn-intelex

  - into your current environment::
    
       conda install scikit-learn-intelex -c intel

.. seealso:: :ref:`system_requirements`

.. _build_from_sources:

Build from Sources
---------------------

Sources are available in |intelex_repo|_.

.. rubric:: Prerequisites

::

    Python version >= 3.6, < 3.10

Configure the build with environment variables
==============================================

::

    SKLEARNEX_VERSION: sets package version

Build |intelex|
===============

To install the package::

    cd <checkout-dir>
    python setup_sklearnex.py install

To install the package in the development mode::

    cd <checkout-dir>
    python setup.py develop

To install scikit-learn-intelex without downloading daal4py::

    cd <checkout-dir>
    python setup_sklearnex.py install --single-version-externally-managed --record=record.txt

To install scikit-learn-intelex without downloading daal4py in the development mode::

    cd <checkout-dir>
    python setup_sklearnex.py develop --no-deps

.. note::
    The ``develop`` mode will not install the package but it will create a ``.egg-link`` in the deployment directory
    back to the project source code directory. That way you can edit the source code and see the changes
    without having to reinstall the package every time you make a small change.

⚠️ Keys ``--single-version-externally-managed`` and ``--no-deps`` are required so that daal4py is not downloaded after installation of |intelex|

.. note::
    ``--single-version-externally-managed`` is an option used for Python packages instructing the setuptools module
    to create a Python package that can be easily managed by the package manager on the host
