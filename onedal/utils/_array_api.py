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
        if isinstance(array, usm_ndarray):
            return dpnp.array(array, copy=False)
        elif isinstance(array, Iterable):
            for i in range(len(array)):
                array[i] = _convert_to_dpnp(array[i])
        return array


def _from_dlpack(data, xp, *args, **kwargs):
    def _one_from_dlpack(data, xp, *args, **kwargs):
        return xp.from_dlpack(data, *args, **kwargs)

    if isinstance(data, Iterable):
        for i in range(len(data)):
            data[i] = _one_from_dlpack(data[i], xp, *args, **kwargs)
        return data
    return _one_from_dlpack(data, xp, *args, **kwargs)


def _is_numpy_namespace(xp):
    """Return True if xp is backed by NumPy."""
    return xp.__name__ in {"numpy", "array_api_compat.numpy", "numpy.array_api"}