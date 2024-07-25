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

"""Tools to support array_api."""

import numpy as np

from .._device_offload import dpnp_available

if dpnp_available:
    import dpnp


def _get_sycl_namespace(*arrays):
    """Get namespace of sycl arrays."""

    # sycl support designed to work regardless of array_api_dispatch sklearn global value
    sycl_type = {type(x): x for x in arrays if hasattr(x, "__sycl_usm_array_interface__")}

    if len(sycl_type) > 1:
        raise ValueError(f"Multiple SYCL types for array inputs: {sycl_type}")

    if sycl_type:
        (X,) = sycl_type.values()

        if hasattr(X, "__array_namespace__"):
            return sycl_type, X.__array_namespace__(), True
        elif dpnp_available and isinstance(X, dpnp.ndarray):
            # convert it to dpctl.tensor with namespace.
            return sycl_type, dpnp, False
        else:
            raise ValueError(f"SYCL type not recognized: {sycl_type}")

    return sycl_type, None, False


# TODO:
# docstring
def get_namespace(*arrays):
    """Get namespace of arrays.

    TBD.

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
    sycl_type, xp, is_array_api_compliant = _get_sycl_namespace(*arrays)

    if sycl_type:
        return xp, is_array_api_compliant
    else:
        return np, True
