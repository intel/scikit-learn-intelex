.. ******************************************************************************
.. * Copyright 2024 Intel Corporation
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

.. _array_api:

#################
Array API support
#################
The `Array API <https://data-apis.org/array-api/latest/>`_ specification defines
a standard API for all array manipulation libraries with a NumPy-like API.
Intel(R) Extension for Scikit-Learn's Array API support requires
`array-api-compat <https://github.com/data-apis/array-api-compat>`__ to be installed.

.. note::
    Currently, only `array-api-strict <https://github.com/data-apis/array-api-strict>`__, `dpctl <https://intelpython.github.io/dpctl/latest/index.html>`__, `dpnp <https://github.com/IntelPython/dpnp>`__ and `numpy <https://numpy.org/>`__ are known to work with estimators.

Support for DPNP and DPCTL
----------------------------------


Example usage
----------------------------------


Support for Array API-compatible inputs
----------------------------------

Planned features
----------------------------------
