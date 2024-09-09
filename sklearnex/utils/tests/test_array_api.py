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
import pytest
from numpy.testing import assert_allclose

from daal4py.sklearn._utils import sklearn_check_version
from onedal.tests.utils._dataframes_support import (
    _convert_to_dataframe,
    get_dataframes_and_queues,
)

array_api_dataframes_and_namespaces = {
    "dpctl": "dpctl.tensor",
}


@pytest.mark.parametrize(
    "dataframe,queue",
    get_dataframes_and_queues(dataframe_filter_="numpy,dpctl", device_filter_="cpu,gpu"),
)
def test_get_namespace_with_config_context(dataframe, queue):
    """Test get_namespace on NumPy ndarrays, DPCtl tensors."""
    from sklearnex import config_context
    from sklearnex.utils._array_api import get_namespace

    # array_api_compat = pytest.importorskip("array_api_strict")

    X_np = np.asarray([[1, 2, 3]])
    X = _convert_to_dataframe(X_np, sycl_queue=queue, target_df=dataframe)

    with config_context(array_api_dispatch=True):
        xp_out, is_array_api_compliant = get_namespace(X)
        assert is_array_api_compliant
        # assert xp_out is array_api_compat.numpy


@pytest.mark.skipif(
    not sklearn_check_version("1.4"),
    reason="array api dispatch requires sklearn 1.4 version",
)
@pytest.mark.parametrize(
    "dataframe,queue",
    get_dataframes_and_queues(dataframe_filter_="dpctl", device_filter_="cpu,gpu"),
)
def test_get_namespace_with_patching(dataframe, queue):
    """Test get_namespace on NumPy ndarrays, DPCtl tensors
    with `patch_sklearn`
    """
    # array_api_compat = pytest.importorskip("array_api_strict")

    from sklearnex import patch_sklearn

    patch_sklearn()

    from sklearn import config_context
    from sklearn.utils._array_api import get_namespace

    X_np = np.asarray([[1, 2, 3]])
    X = _convert_to_dataframe(X_np, sycl_queue=queue, target_df=dataframe)

    with config_context(array_api_dispatch=True):
        xp_out, is_array_api_compliant = get_namespace(X)
        assert is_array_api_compliant
        assert xp_out.__name__ == array_api_dataframes_and_namespaces[dataframe]


@pytest.mark.skipif(
    not sklearn_check_version("1.4"),
    reason="array api dispatch requires sklearn 1.4 version",
)
@pytest.mark.parametrize(
    "dataframe,queue",
    get_dataframes_and_queues(dataframe_filter_="dpctl,dpnp", device_filter_="cpu,gpu"),
)
def test_convert_to_numpy_with_patching(dataframe, queue):
    """Test get_namespace on NumPy ndarrays, DPCtl tensors
    with `patch_sklearn`
    """
    # array_api_compat = pytest.importorskip("array_api_strict")

    from sklearnex import patch_sklearn

    patch_sklearn()

    from sklearn import config_context
    from sklearn.utils._array_api import _convert_to_numpy, get_namespace

    X_np = np.asarray([[1, 2, 3]])
    X = _convert_to_dataframe(X_np, sycl_queue=queue, target_df=dataframe)
    xp, _ = get_namespace(X)

    with config_context(array_api_dispatch=True):
        x_np = _convert_to_numpy(X, xp)
        assert type(X_np) == type(x_np)
        assert_allclose(X_np, x_np)


@pytest.mark.skipif(
    not sklearn_check_version("1.4"),
    reason="array api dispatch requires sklearn 1.4 version",
)
@pytest.mark.parametrize(
    "dataframe,queue",
    get_dataframes_and_queues(dataframe_filter_="dpctl", device_filter_="cpu,gpu"),
)
def test_check_array_with_patching(dataframe, queue):
    """Test get_namespace on NumPy ndarrays, DPCtl tensors
    with `patch_sklearn`
    """
    # array_api_compat = pytest.importorskip("array_api_strict")

    from sklearnex import patch_sklearn

    patch_sklearn()

    from sklearn import config_context
    from sklearn.utils import check_array
    from sklearn.utils._array_api import _convert_to_numpy, get_namespace

    X_np = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    xp, _ = get_namespace(X_np)
    X_df = _convert_to_dataframe(X_np, sycl_queue=queue, target_df=dataframe)

    with config_context(array_api_dispatch=True):
        X_df_res = check_array(X_df, accept_sparse="csr", dtype=[xp.float64, xp.float32])
        assert type(X_df) == type(X_df_res)
        assert_allclose(_convert_to_numpy(X_df, xp), _convert_to_numpy(X_df_res, xp))
