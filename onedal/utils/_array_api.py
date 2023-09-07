# ===============================================================================
# Copyright 2023 Intel Corporation
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
# ===============================================================================

import numpy as np

try:
    import dpnp

    dpnp_available = True
except ImportError:
    dpnp_available = False

try:
    import dpctl
    import dpctl.tensor as dpt

    dpctl_available = True
except ImportError:
    dpctl_available = False


def as_dpnp_ndarray(X, copy=False, queue=None, *args, **kwargs):
    if dpnp_available and isinstance(X, dpnp.ndarray):
        return X
    elif dpctl_available and isinstance(X, dpt.usm_ndarray):
        return dpnp.array(X, copy=copy)
    else:
        return dpnp.array(X, sycl_queue=queue, *args, **kwargs)


def array(X, copy=False, queue=None, *args, **kwargs):
    pass


def _as_numpy(obj, *args, **kwargs):
    if dpnp_available and isinstance(obj, dpnp.ndarray):
        return obj.asnumpy(*args, **kwargs)
    if dpctl_available and isinstance(obj, dpt.usm_ndarray):
        return dpt.to_numpy(obj, *args, **kwargs)
    return np.asarray(obj, *args, **kwargs)


def ndarray_take(obj, *args, **kwargs):
    if (
        isinstance(X, np.ndarray)
        or (dpnp_available and isinstance(obj, dpnp.ndarray))
        or (dpctl_available and isinstance(obj, dpt.usm_ndarray))
    ):
        return obj.take(*args, **kwargs)
