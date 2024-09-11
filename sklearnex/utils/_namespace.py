# ==============================================================================
# Copyright 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np

from daal4py.sklearn._utils import sklearn_check_version

from .._device_offload import dpnp_available

if sklearn_check_version("1.2"):
    from sklearn.utils._array_api import get_namespace as sklearn_get_namespace

if dpnp_available:
    import dpnp


def get_namespace(*arrays):
    """Get namespace of arrays.

    Introspect `arrays` arguments and return their common Array API
    compatible namespace object, if any. NumPy 1.22 and later can
    construct such containers using the `numpy.array_api` namespace
    for instance.

    This function will return the namespace of SYCL-related arrays
    which define the __sycl_usm_array_interface__ attribute
    regardless of array_api support, the configuration of
    array_api_dispatch, or scikit-learn version.

    See: https://numpy.org/neps/nep-0047-array-api-standard.html

    If `arrays` are regular numpy arrays, an instance of the
    `_NumPyApiWrapper` compatibility wrapper is returned instead.

    Namespace support is not enabled by default. To enabled it
    call:

      sklearn.set_config(array_api_dispatch=True)

    or:

      with sklearn.config_context(array_api_dispatch=True):
          # your code here

    Otherwise an instance of the `_NumPyApiWrapper`
    compatibility wrapper is always returned irrespective of
    the fact that arrays implement the `__array_namespace__`
    protocol or not.

    Parameters
    ----------
    *arrays : array objects
        Array objects.

    Returns
    -------
    namespace : module
        Namespace shared by array objects.

    is_array_api : bool
        True of the arrays are containers that implement the Array API spec.
    """

    # sycl support designed to work regardless of array_api_dispatch sklearn global value
    sycl_type = {type(x): x for x in arrays if hasattr(x, "__sycl_usm_array_interface__")}

    if len(sycl_type) > 1:
        raise ValueError(f"Multiple SYCL types for array inputs: {sycl_type}")

    if sycl_type:

        (X,) = sycl_type.values()

        if hasattr(X, "__array_namespace__"):
            return X.__array_namespace__(), True
        elif dpnp_available and isinstance(X, dpnp.ndarray):
            return dpnp, False
        else:
            raise ValueError(f"SYCL type not recognized: {sycl_type}")

    elif sklearn_check_version("1.2"):
        return sklearn_get_namespace(*arrays)
    else:
        return np, False
