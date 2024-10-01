# ==============================================================================
# Copyright 2021 Intel Corporation
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
from sklearn.datasets import make_blobs

from onedal import _backend, _is_dpc_backend
from onedal._device_offload import dpctl_available, dpnp_available
from onedal.datatypes import from_table, to_table

# TODO:
# re-impl and use from_table, to_table instead.
from onedal.datatypes._data_conversion import convert_one_from_table, convert_one_to_table
from onedal.datatypes.tests.common import _assert_sua_iface_fields, _assert_tensor_attr
from onedal.primitives import linear_kernel
from onedal.tests.utils._dataframes_support import (
    _convert_to_dataframe,
    get_dataframes_and_queues,
)
from onedal.tests.utils._device_selection import get_queues
from onedal.utils._array_api import _get_sycl_namespace

if dpctl_available:
    import dpctl.tensor as dpt

if dpnp_available:
    import dpnp


data_shapes = [
    pytest.param((1000, 100), id="(1000, 100)"),
    pytest.param((2000, 50), id="(2000, 50)"),
]

ORDER_DICT = {"F": np.asfortranarray, "C": np.ascontiguousarray}


if _is_dpc_backend:
    from daal4py.sklearn._utils import get_dtype
    from onedal.cluster.dbscan import BaseDBSCAN
    from onedal.common._policy import _get_policy

    # TODO:
    # use  from_table, to_table.
    from onedal.datatypes._data_conversion import (
        convert_one_from_table,
        convert_one_to_table,
    )

    class DummyEstimatorWithTableConversions:

        def fit(self, X, y=None):
            policy = _get_policy(X.sycl_queue, None)
            bs_DBSCAN = BaseDBSCAN()
            types = [np.float32, np.float64]
            if get_dtype(X) not in types:
                X = X.astype(np.float64)
            dtype = get_dtype(X)
            params = bs_DBSCAN._get_onedal_params(dtype)
            X_table = convert_one_to_table(X, True)
            # TODO:
            # check other candidates for the dummy base OneDAL func.
            # OneDAL backend func is needed to check result table checks.
            result = _backend.dbscan.clustering.compute(
                policy, params, X_table, convert_one_to_table(None)
            )
            result_responses_table = result.responses
            sua_iface, xp, _ = _get_sycl_namespace(X)
            result_responses_df = convert_one_from_table(
                result_responses_table, sua_iface=sua_iface, xp=xp
            )
            return X_table, result_responses_table, result_responses_df

else:

    class DummyEstimatorWithTableConversions:
        pass


def _test_input_format_c_contiguous_numpy(queue, dtype):
    rng = np.random.RandomState(0)
    x_default = np.array(5 * rng.random_sample((10, 4)), dtype=dtype)

    x_numpy = np.asanyarray(x_default, dtype=dtype, order="C")
    assert x_numpy.flags.c_contiguous
    assert not x_numpy.flags.f_contiguous
    assert not x_numpy.flags.fnc

    expected = linear_kernel(x_default, queue=queue)
    result = linear_kernel(x_numpy, queue=queue)
    assert_allclose(expected, result)


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_input_format_c_contiguous_numpy(queue, dtype):
    if queue and queue.sycl_device.is_gpu:
        pytest.skip("Sporadic failures on GPU sycl_queue.")
    _test_input_format_c_contiguous_numpy(queue, dtype)


def _test_input_format_f_contiguous_numpy(queue, dtype):
    rng = np.random.RandomState(0)
    x_default = np.array(5 * rng.random_sample((10, 4)), dtype=dtype)

    x_numpy = np.asanyarray(x_default, dtype=dtype, order="F")
    assert not x_numpy.flags.c_contiguous
    assert x_numpy.flags.f_contiguous
    assert x_numpy.flags.fnc

    expected = linear_kernel(x_default, queue=queue)
    result = linear_kernel(x_numpy, queue=queue)
    assert_allclose(expected, result)


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_input_format_f_contiguous_numpy(queue, dtype):
    if queue and queue.sycl_device.is_gpu:
        pytest.skip("Sporadic failures on GPU sycl_queue.")
    _test_input_format_f_contiguous_numpy(queue, dtype)


def _test_input_format_c_not_contiguous_numpy(queue, dtype):
    rng = np.random.RandomState(0)
    x_default = np.array(5 * rng.random_sample((10, 4)), dtype=dtype)

    dummy_data = np.insert(x_default, range(1, x_default.shape[1]), 8, axis=1)
    x_numpy = dummy_data[:, ::2]

    assert_allclose(x_numpy, x_default)

    assert not x_numpy.flags.c_contiguous
    assert not x_numpy.flags.f_contiguous
    assert not x_numpy.flags.fnc

    expected = linear_kernel(x_default, queue=queue)
    result = linear_kernel(x_numpy, queue=queue)
    assert_allclose(expected, result)


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_input_format_c_not_contiguous_numpy(queue, dtype):
    if queue and queue.sycl_device.is_gpu:
        pytest.skip("Sporadic failures on GPU sycl_queue.")
    _test_input_format_c_not_contiguous_numpy(queue, dtype)


def _test_input_format_c_contiguous_pandas(queue, dtype):
    pd = pytest.importorskip("pandas")
    rng = np.random.RandomState(0)
    x_default = np.array(5 * rng.random_sample((10, 4)), dtype=dtype)

    x_numpy = np.asanyarray(x_default, dtype=dtype, order="C")
    assert x_numpy.flags.c_contiguous
    assert not x_numpy.flags.f_contiguous
    assert not x_numpy.flags.fnc
    x_df = pd.DataFrame(x_numpy)

    expected = linear_kernel(x_df, queue=queue)
    result = linear_kernel(x_numpy, queue=queue)
    assert_allclose(expected, result)


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_input_format_c_contiguous_pandas(queue, dtype):
    if queue and queue.sycl_device.is_gpu:
        pytest.skip("Sporadic failures on GPU sycl_queue.")
    _test_input_format_c_contiguous_pandas(queue, dtype)


def _test_input_format_f_contiguous_pandas(queue, dtype):
    pd = pytest.importorskip("pandas")
    rng = np.random.RandomState(0)
    x_default = np.array(5 * rng.random_sample((10, 4)), dtype=dtype)

    x_numpy = np.asanyarray(x_default, dtype=dtype, order="F")
    assert not x_numpy.flags.c_contiguous
    assert x_numpy.flags.f_contiguous
    assert x_numpy.flags.fnc
    x_df = pd.DataFrame(x_numpy)

    expected = linear_kernel(x_df, queue=queue)
    result = linear_kernel(x_numpy, queue=queue)
    assert_allclose(expected, result)


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_input_format_f_contiguous_pandas(queue, dtype):
    if queue and queue.sycl_device.is_gpu:
        pytest.skip("Sporadic failures on GPU sycl_queue.")
    _test_input_format_f_contiguous_pandas(queue, dtype)


def _test_conversion_to_table(dtype):
    np.random.seed()
    if dtype in [np.int32, np.int64]:
        x = np.random.randint(0, 10, (15, 3), dtype=dtype)
    else:
        x = np.random.uniform(-2, 2, (18, 6)).astype(dtype)
    x_table = to_table(x)
    x2 = from_table(x_table)
    assert x.dtype == x2.dtype
    assert np.array_equal(x, x2)


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
def test_conversion_to_table(dtype):
    _test_conversion_to_table(dtype)


# TODO:
# rename test suit.
@pytest.mark.parametrize(
    "dataframe,queue", get_dataframes_and_queues("dpctl,dpnp", "cpu, gpu")
)
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_input_sua_iface_zero_copy(dataframe, queue, order, dtype):
    # TODO:
    # investigate in the same PR.
    if dataframe in "dpnp":
        pytest.skip("Some bug with Sycl.Queue for DPNP inputs.")
    # TODO:
    # add description to the test.
    rng = np.random.RandomState(0)
    X_default = np.array(5 * rng.random_sample((10, 59)), dtype=dtype)

    X_np = np.asanyarray(X_default, dtype=dtype, order=order)

    X_dp = _convert_to_dataframe(X_np, sycl_queue=queue, target_df=dataframe)

    sua_iface, X_dp_namespace, _ = _get_sycl_namespace(X_dp)

    X_table = convert_one_to_table(X_dp, sua_iface=sua_iface)
    # TODO:
    # investigate in the same PR skip_syclobj WO.
    _assert_sua_iface_fields(X_dp, X_table, skip_syclobj=True)

    X_dp_from_table = convert_one_from_table(
        X_table, sua_iface=sua_iface, xp=X_dp_namespace
    )
    # TODO:
    # investigate in the same PR skip_syclobj WO.
    _assert_sua_iface_fields(X_table, X_dp_from_table, skip_syclobj=True)
    _assert_tensor_attr(X_dp, X_dp_from_table, order)


# TODO:
# rename test suit.
@pytest.mark.skipif(
    not _is_dpc_backend,
    reason="__sycl_usm_array_interface__ support requires DPC backend.",
)
@pytest.mark.parametrize(
    "dataframe,queue", get_dataframes_and_queues("dpctl, dpnp", "cpu, gpu")
)
@pytest.mark.parametrize("order", ["F", "C"])
@pytest.mark.parametrize("data_shape", data_shapes)
def test_table_conversions(dataframe, queue, order, data_shape):
    # TODO:
    # Description for the test.
    if queue.sycl_device.is_cpu:
        pytest.skip("OneDAL returns None sycl queue for CPU sycl queue inputs.")
    # TODO:
    # investigate in the same PR.
    if dataframe in "dpnp":
        pytest.skip("Some bug with Sycl.Queue for DPNP inputs.")

    n_samples, n_features = data_shape
    X, y = make_blobs(
        n_samples=n_samples, centers=3, n_features=n_features, random_state=0
    )

    X = ORDER_DICT[order](X)

    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    alg = DummyEstimatorWithTableConversions()
    X_table, result_responses_table, result_responses_df = alg.fit(X)

    assert hasattr(X_table, "__sycl_usm_array_interface__")
    assert hasattr(result_responses_table, "__sycl_usm_array_interface__")
    assert hasattr(result_responses_df, "__sycl_usm_array_interface__")
    assert hasattr(X, "__sycl_usm_array_interface__")
    # TODO:
    # investigate in the same PR skip_syclobj and skip_data_1 WO.
    _assert_sua_iface_fields(X, X_table, skip_syclobj=True, skip_data_1=True)
    # TODO:
    # investigate in the same PR skip_syclobj and skip_data_1 WO.
    _assert_sua_iface_fields(
        result_responses_df, result_responses_table, skip_syclobj=True, skip_data_1=True
    )
    assert X.sycl_queue == result_responses_df.sycl_queue
    if order == "F":
        assert X.flags.f_contiguous == result_responses_df.flags.f_contiguous
    else:
        assert X.flags.c_contiguous == result_responses_df.flags.c_contiguous
    # 1D output expected to have the same c_contiguous and f_contiguous flag values.
    assert (
        result_responses_df.flags.c_contiguous == result_responses_df.flags.f_contiguous
    )


# TODO:
# def test_wrong_inputs_for_sua_iface_conversion
