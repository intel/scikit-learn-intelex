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

from collections.abc import Iterable

try:
    from dpctl.tensor import usm_ndarray

    dpctl_available = True
except ImportError:
    dpctl_available = False

try:
    import dpnp

    dpnp_available = True
except ImportError:
    dpnp_available = False


if dpnp_available:
    import dpnp

    def _convert_to_dpnp(array):
        """Converted input object to dpnp.ndarray format."""
        if isinstance(array, usm_ndarray):
            return dpnp.array(array, copy=False)
        elif isinstance(array, Iterable):
            for i in range(len(array)):
                array[i] = _convert_to_dpnp(array[i])
        return array


def _asarray(data, xp, *args, **kwargs):
    def _one_asarray(data, xp, *args, **kwargs):
        if data is not None:
            return xp.asarray(data, *args, **kwargs)

    if hasattr(data, "__array_namespace__"):
        return _one_asarray(data, xp, *args, **kwargs)
    elif isinstance(data, Iterable):
        for i in range(len(data)):
            data[i] = _one_asarray(data[i], xp, *args, **kwargs)
    return data


def _is_numpy_namespace(xp):
    """Return True if xp is backed by NumPy."""
    return xp.__name__ in {"numpy", "array_api_compat.numpy", "numpy.array_api"}


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
            return sycl_type, dpnp, False
        else:
            raise ValueError(f"SYCL type not recognized: {sycl_type}")

    return sycl_type, None, False
