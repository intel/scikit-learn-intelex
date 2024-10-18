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
Intel(R) Extension for Scikit-Learn doesn't require
`array-api-compat <https://github.com/data-apis/array-api-compat>`__ to be installed for
functional support of Array API.
In the current implementation, the functional support of array api is just part of functional
support of different array or dataframe inputs and tries to leave presecion of the input and
output data formats. Any array API input data provided will be converted to host numpy.ndarrays
and all internal manipulations with data will be done with these representations of the input
data. Work with DPNP.ndarray and DPCTL usm_ndarray has specifics, that is described in the
relevant section of this document.

The main goal is to save preeservance of returned inputs data formats for the outputs.

.. note::
    Currently, only `array-api-strict <https://github.com/data-apis/array-api-strict>`__,
    `dpctl <https://intelpython.github.io/dpctl/latest/index.html>`__, `dpnp <https://github.com/IntelPython/dpnp>`__
    and `numpy <https://numpy.org/>`__ are known to work with estimators.
.. note::
    Stock Scikit-learn’s Array API support requires `array-api-compat <https://github.com/data-apis/array-api-compat>`__ to be installed.


Support for DPNP and DPCTL
----------------------------------
The functional support of input data for sklearnex estimators also extended for dpnp.ndarrays,
which are not Array API compliant, and dpctl usm_ndarray, which are array API compliant except
`linalg` module. DPNP and DPCTL usm_ndarray sycl context used for scikit-learn-intelex device
offloading.

.. note::
    Current support for DPNP and DPCTL usm_ndarray doesn't guarantee zero-copy and non-data movement from gpu device
    to host.

DPCTL or DPNP inputs doesn't require to use ```config_context(target_offload=device)```.
`sklearnex` will use input usm_ndarray sycl context for device offloading.

.. note::
    If ```config_context(target_offload=device)``` enabled for DPCTL or DPNP inputs and device is as a string
    for the device primarily used to perform computations, returned output will have different sycl queue for
    the device. `target_offload` provided with `config_context` will be used to perform computations.

.. automodule:: examples/sklearnex/random_forest_regressor_dpnp.py
  :members:

Example usage
----------------------------------
TBD


Support for Array API-compatible inputs
----------------------------------
All patched estimators, metrics, tools and non-scikit-learn estimators functionally support Array API.
Functionally all input and output data have the same data format type.

Stock scikit-learn uses ```config_context(array_api_dispatch=True)``` for enabling Array API `support <https://scikit-learn.org/1.5/modules/array_api.html>`__.
If `array_api_dispatch` enabled and array api is supported for the stock scikit-learn, then raw inputs are used for the fallback.

.. note::
    Stock scikit-learn doesn't support DPCTL usm_ndarray or DPNP ndarray inputs, host numpy copies
    of the inputs data will be used for the fallback cases. This fallback condition will be changed
    in the next versions, depending on the support of Array API standard by DPNP and DPCTL and
    reimplementation of `sklearnex` estimators via Array API.


Planned features
----------------------------------
TBD
