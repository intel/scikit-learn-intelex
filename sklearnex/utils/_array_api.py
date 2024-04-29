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
import itertools

import numpy as np

from daal4py.sklearn._utils import sklearn_check_version

from .._device_offload import dpctl_available, dpnp_available

# import math
# from functools import wraps


if sklearn_check_version("1.2"):
    from sklearn.utils._array_api import _convert_to_numpy as _sklearn_convert_to_numpy
    from sklearn.utils._array_api import get_namespace as sklearn_get_namespace
    from sklearn.utils._array_api import (
        yield_namespace_device_dtype_combinations as sklearn_yield_namespace_device_dtype_combinations,
    )

if dpctl_available:
    import dpctl.tensor as dpt

if dpnp_available:
    import dpnp


def _convert_to_numpy(array, xp):
    """Convert X into a NumPy ndarray on the CPU."""
    xp_name = xp.__name__

    if dpctl_available and xp_name in {
        "dpctl.tensor",
    }:
        return dpt.to_numpy(array)
    else:
        return _sklearn_convert_to_numpy(array, xp)


def yield_namespace_device_dtype_combinations():
    """Yield supported namespace, device, dtype tuples for testing.

    Use this to test that an estimator works with all combinations.

    Returns
    -------
    array_namespace : str
        The name of the Array API namespace.

    device : str
        The name of the device on which to allocate the arrays. Can be None to
        indicate that the default value should be used.

    dtype_name : str
        The name of the data type to use for arrays. Can be None to indicate
        that the default value should be used.
    """
    for array_namespace in [
        # The following is used to test the array_api_compat wrapper when
        # array_api_dispatch is enabled: in particular, the arrays used in the
        # tests are regular numpy arrays without any "device" attribute.
        "numpy",
        # Stricter NumPy-based Array API implementation. The
        # numpy.array_api.Array instances always a dummy "device" attribute.
        "numpy.array_api",
        "cupy",
        "cupy.array_api",
        "torch",
        "dpctl.tensor",
    ]:
        if array_namespace == "torch":
            for device, dtype in itertools.product(
                ("cpu", "cuda"), ("float64", "float32")
            ):
                yield array_namespace, device, dtype
            yield array_namespace, "mps", "float32"
        else:
            yield array_namespace, None, None
