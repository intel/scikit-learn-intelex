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

import pytest
import scipy.sparse as sp
from sklearn import get_config

try:
    import dpctl
    import dpctl.tensor as dpt

    dpctl_available = True
except ImportError:
    dpctl_available = False

try:
    import dpnp

    dpnp_available = True
except ImportError:
    dpnp_available = False

try:
    # This should be lazy imported in the
    # future along with other popular
    # array_api libraries when testing
    # GPU-no-copy.
    import array_api_strict

    # Run check if "array_api_dispatch" is configurable
    array_api_enabled = lambda: get_config()["array_api_dispatch"]
    array_api_enabled()
    array_api_modules = {"array_api": array_api_strict}


except (ImportError, KeyError):
    array_api_enabled = lambda: False
    array_api_modules = {}


import numpy as np
import pandas as pd

from onedal.tests.utils._device_selection import get_queues


def get_dataframes_and_queues(
    dataframe_filter_="numpy,pandas,dpnp,dpctl", device_filter_="cpu,gpu"
):
    """Get supported dataframes for testing.

    This is meant to be used for testing purposes only.

    Parameters
    ----------
    dataframe_filter_ : str, default="numpy,pandas,dpnp,dpctl"
        Configure output pytest.params for the certain dataframe formats.
    device_filter_ : str, default="cpu,gpu"
        Configure output pytest.params with certain sycl queue for the dataframe,
        where it is applicable.

    Returns
    -------
    list[pytest.param]
        The list of pytest params, included dataframe name (str),
        sycl queue, if applicable for the test case, and test
        case id (str).

    Notes
    -----
        Do not use filters for the test cases disabling. Use `pytest.skip`
        or `pytest.xfail` instead.

    See Also
    --------
    _convert_to_dataframe : Converted input object to certain dataframe format.
    """
    dataframes_and_queues = []

    if "numpy" in dataframe_filter_:
        dataframes_and_queues.append(pytest.param("numpy", None, id="numpy"))
    if "pandas" in dataframe_filter_:
        dataframes_and_queues.append(pytest.param("pandas", None, id="pandas"))

    def get_df_and_q(dataframe: str):
        df_and_q = []
        for queue in get_queues(device_filter_):
            if queue:
                id = "{}-{}".format(dataframe, queue.id)
                df_and_q.append(pytest.param(dataframe, queue.values[0], id=id))
        return df_and_q

    if dpctl_available and "dpctl" in dataframe_filter_:
        dataframes_and_queues.extend(get_df_and_q("dpctl"))
    if dpnp_available and "dpnp" in dataframe_filter_:
        dataframes_and_queues.extend(get_df_and_q("dpnp"))
    if "array_api" in dataframe_filter_ or array_api_enabled():
        dataframes_and_queues.append(pytest.param("array_api", None, id="array_api"))

    return dataframes_and_queues


def _as_numpy(obj, *args, **kwargs):
    if dpnp_available and isinstance(obj, dpnp.ndarray):
        return obj.asnumpy(*args, **kwargs)
    if dpctl_available and isinstance(obj, dpt.usm_ndarray):
        return dpt.to_numpy(obj, *args, **kwargs)
    if isinstance(obj, pd.DataFrame) or isinstance(obj, pd.Series):
        return obj.to_array(*args, **kwargs)
    if sp.issparse(obj):
        return obj.toarray(*args, **kwargs)
    return np.asarray(obj, *args, **kwargs)


def _convert_to_dataframe(obj, sycl_queue=None, target_df=None, *args, **kwargs):
    """Converted input object to certain dataframe format."""
    if target_df is None:
        return obj
    elif target_df == "numpy":
        # Numpy ndarray.
        # `sycl_queue` arg is ignored.
        return np.asarray(obj, *args, **kwargs)
    # Pandas Dataframe
    elif target_df == "pandas":
        if (
            "dtype" in kwargs
            and hasattr(obj, "astype")
            and np.issubdtype(kwargs["dtype"], np.integer)
        ):
            # Pandas float to int not allowed
            obj = obj.astype(kwargs["dtype"])
        if hasattr(obj, "ndim") and obj.ndim == 1:
            return pd.Series(obj, *args, **kwargs)
        else:
            return pd.DataFrame(obj, *args, **kwargs)
    # DPNP ndarray.
    elif target_df == "dpnp":
        return dpnp.asarray(
            obj, usm_type="device", sycl_queue=sycl_queue, *args, **kwargs
        )
    elif target_df == "dpctl":
        # DPCtl tensor.
        return dpt.asarray(obj, usm_type="device", sycl_queue=sycl_queue, *args, **kwargs)
    elif target_df in array_api_modules:
        # use dpctl to define gpu devices via queues and
        # move data to the device. This is necessary as
        # the standard for defining devices is
        # purposefully not defined in the array_api
        # standard, but maintaining data on a device
        # using the method `from_dlpack` is.
        xp = array_api_modules[target_df]
        return xp.from_dlpack(
            _convert_to_dataframe(
                obj, sycl_queue=sycl_queue, target_df="dpctl", *args, **kwargs
            )
        )

    raise RuntimeError("Unsupported dataframe conversion")
