.. ******************************************************************************
.. * Copyright 2021 Intel Corporation
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

.. _system_requirements:

###################
System requirements
###################

This page provides details about hardware, operating system, and software prerequisites for the Intel® Extension for Scikit-learn.

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
     - ✔️
   * - Windows
     - ✔️
     - ✔️
     - ✔️
     - ✔️
   * - macOS
     - ✔️
     - ✔️
     - ✔️
     - ✔️

.. note::
    It supports CPUs and GPUs

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
   * - macOS
     - ✔️
     - ✔️
     - ✔️
     - ✔️

.. note::
    It supports only CPUs.
    Recommended for conda users by default

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
     - ✔️
   * - Windows
     - ✔️
     - ✔️
     - ✔️
     - ✔️
   * - macOS
     - ✔️
     - ✔️
     - ✔️
     - ✔️

.. note::
    It supports CPUs and GPUs.
    Recommended for conda users who use other components from Intel(R) Distribution for Python

System Requirements
-------------------

For CPU users
-------------

**Operating systems**

- **Linux***: the last two versions of popular Linux systems
- **Windows*** and **Windows* Server**: the last two versions 
- **macOS***: the last two versions 

**Hardware platforms**

- All processors with x86 architecture

.. note::
    Your processor must support at least one of SSE2. AVX, AVX2, AVX512 instruction set

.. note::
    ARM* architecture is not supported

.. note::
    Intel® processors provide better performance then other CPUs.
    Read more about hardware comparison in our :ref:`blogs <blogs>`

For GPU users
-------------

.. warning::
    For users using accelerators, please refer to the DPC++ compiler system
    requirements `here <https://software.intel.com/content/www/us/en/develop/articles/intel-oneapi-dpcpp-system-requirements.html>`_

**Operating systems**

- **Linux***: the last two versions of popular Linux systems
- **Windows*** and **Windows* Server**: the last two versions

**Hardware platforms**

- All Intel® integrated and discrete GPUs
- `Intel® GPU drivers <https://www.intel.com/content/www/us/en/develop/articles/installation-guide-for-intel-oneapi-toolkits.html#installGPUdriver>`__
