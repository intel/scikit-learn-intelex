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

.. note::
    Intel® Extension for Scikit-learn may work on other hardware, operating systems, and with other configurations, but that was not tested.
    You can find our blogs about hardware comparison :ref:`here <blogs>`.

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
   * - OsX
     - ✔️
     - ✔️
     - ✔️
     - ✔️

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
    It supports Intel CPU and GPU except on OsX.
    Recommended for conda users who use other components from Intel(R) Distribution for Python.

System Requirements for Data Parallel C++ (DPC++)
-------------------------------------------------

For users using the DPC++ compiler, please refer to the DPC++ compiler system
requirements `here <https://software.intel.com/content/www/us/en/develop/articles/intel-oneapi-dpcpp-system-requirements.html>`_.

Operating systems
-----------------
- Linux*
- Redhat Enterprise Linux (RHEL)* 7, 8
- Ubuntu* 18.04 LTS, 20.04 LTS
- Windows*
- Windows* Server 2019
- Windows* 10
- macOS*
- macOS* 10.15
- macOS* 11

Supported Hardware Platforms
----------------------------

**CPU**

- Intel Atom® Processors
- Intel® Core™ Processor Family
- Intel® Xeon® Processor Family
- Intel® Xeon® Scalable Performance Processor Family

**Accelerators**

- Intel® HD Graphics
- Intel® UHD Graphics for 9th, 10th and 11th Gen Intel® Processors
- Intel® Iris® Plus Graphics
- Intel® Iris® Xe Graphics
- Intel® Iris® Xe Max Graphics
- Intel® Iris® Graphics
- Intel® Iris® Pro Graphics

Software Requirements
---------------------

**For DPC++**

*Linux**

- GNU* GCC v7.0 or higher
- Intel® oneAPI DPC++/C++ Compiler 2021.1 and later
- Intel® oneAPI Threading Building Blocks (oneTBB) 2021.1 and later
- `Intel GPU drivers <https://www.intel.com/content/www/us/en/develop/articles/installation-guide-for-intel-oneapi-toolkits.html#installGPUdriver>`__  for Intel GPU development

*Windows**

- Intel® oneAPI DPC++/C++ Compiler 2021.1 and later
- Intel® oneAPI Threading Building Blocks (oneTBB) 2021.1 and later
- `Intel GPU drivers <https://www.intel.com/content/www/us/en/develop/articles/installation-guide-for-intel-oneapi-toolkits.html#installGPUdriver>`__ for Intel GPU development

**For C/C++**

*Linux**

- Intel® C++ Compiler Classic 2021.1 and later
- Intel® oneAPI DPC++/C++ Compiler 2021.1 and later
- Intel® C++ compiler v19.0 and v19.1
- GNU* Compiler Collection 5.x and later
- PGI* Compiler 19.10 and 20.4
- Intel® oneAPI Threading Building Blocks (oneTBB) 2021.1 and later

*Windows**

- Intel® C/C++ Compiler Classic 2021.1 and later
- Intel® oneAPI DPC++/C++ Compiler 2021.1 and later
- Intel® C++ compiler v19.0 and v19.1
- PGI* Compiler 19.10 and 20.4
- Intel® oneAPI Threading Building Blocks (oneTBB) 2021.1 and later

*macOS**

- Xcode* 11, 12
- Intel® C/C++ Compiler Classic 2021.1 and later
- Intel® oneAPI DPC++/C++ Compiler 2021.1 and later
- Intel® C++ compiler v19.1
- Intel® oneAPI Threading Building Blocks (oneTBB) 2021.1 and later
