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

import time

import numpy as np
import numpy.random as rand
import pytest

from daal4py.sklearn._utils import sklearn_check_version
from onedal.tests.utils._dataframes_support import (
    _convert_to_dataframe,
    get_dataframes_and_queues,
)
from sklearnex import config_context
from sklearnex.tests.utils import DummyEstimator, gen_dataset
from sklearnex.utils.validation import _check_sample_weight, validate_data


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "shape",
    [
        [16, 2048],
        [
            2**16 + 3,
        ],
        [1000, 1000],
    ],
)
@pytest.mark.parametrize("ensure_all_finite", ["allow-nan", True])
def test_sum_infinite_actually_finite(dtype, shape, ensure_all_finite):
    est = DummyEstimator()
    X = np.empty(shape, dtype=dtype)
    X.fill(np.finfo(dtype).max)
    X = np.atleast_2d(X)
    X_array = validate_data(est, X, ensure_all_finite=ensure_all_finite)
    assert type(X_array) == type(X)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "shape",
    [
        [16, 2048],
        [
            2**16 + 3,
        ],
        [1000, 1000],
    ],
)
@pytest.mark.parametrize("ensure_all_finite", ["allow-nan", True])
@pytest.mark.parametrize("check", ["inf", "NaN", None])
@pytest.mark.parametrize("seed", [0, int(time.time())])
@pytest.mark.parametrize("dataframe, queue", get_dataframes_and_queues())
def test_validate_data_random_location(
    dataframe, queue, dtype, shape, ensure_all_finite, check, seed
):
    est = DummyEstimator()
    rand.seed(seed)
    X = rand.uniform(high=np.finfo(dtype).max, size=shape).astype(dtype)

    if check:
        loc = rand.randint(0, X.size - 1)
        X.reshape((-1,))[loc] = float(check)

    # column heavy pandas inputs are very slow in sklearn's check_array even without
    # the finite check, just transpose inputs to guarantee fast processing in tests
    X = _convert_to_dataframe(
        np.atleast_2d(X).T,
        target_df=dataframe,
        sycl_queue=queue,
    )

    allow_nan = ensure_all_finite == "allow-nan"
    if check is None or (allow_nan and check == "NaN"):
        validate_data(est, X, ensure_all_finite=ensure_all_finite)
    else:
        type_err = "infinity" if allow_nan else "NaN, infinity"
        msg_err = f"Input X contains {type_err}."
        with pytest.raises(ValueError, match=msg_err):
            validate_data(est, X, ensure_all_finite=ensure_all_finite)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("ensure_all_finite", ["allow-nan", True])
@pytest.mark.parametrize("check", ["inf", "NaN", None])
@pytest.mark.parametrize("seed", [0, int(time.time())])
@pytest.mark.parametrize("dataframe, queue", get_dataframes_and_queues())
def test_validate_data_random_shape_and_location(
    dataframe, queue, dtype, ensure_all_finite, check, seed
):
    est = DummyEstimator()
    lb, ub = 32768, 1048576  # lb is a patching condition, ub 2^20
    rand.seed(seed)
    X = rand.uniform(high=np.finfo(dtype).max, size=rand.randint(lb, ub)).astype(dtype)

    if check:
        loc = rand.randint(0, X.size - 1)
        X[loc] = float(check)

    X = _convert_to_dataframe(
        np.atleast_2d(X).T,
        target_df=dataframe,
        sycl_queue=queue,
    )

    allow_nan = ensure_all_finite == "allow-nan"
    if check is None or (allow_nan and check == "NaN"):
        validate_data(est, X, ensure_all_finite=ensure_all_finite)
    else:
        type_err = "infinity" if allow_nan else "NaN, infinity"
        msg_err = f"Input X contains {type_err}."
        with pytest.raises(ValueError, match=msg_err):
            validate_data(est, X, ensure_all_finite=ensure_all_finite)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("check", ["inf", "NaN", None])
@pytest.mark.parametrize(
    "array_api_dispatch", [True, False] if sklearn_check_version("1.2") else [False]
)
@pytest.mark.parametrize("seed", [0, int(time.time())])
@pytest.mark.parametrize(
    "dataframe, queue",
    get_dataframes_and_queues(
        "numpy,pandas" + ("dpctl,array_api" if sklearn_check_version("1.2") else "")
    ),
)
def test__check_sample_weights_random_shape_and_location(
    dataframe, queue, dtype, array_api_dispatch, check, seed
):
    # This testing assumes that array api inputs to validate_data will only occur
    # with sklearn array_api support which began in sklearn 1.2. This would assume
    # that somewhere upstream of the validate_data call, a data conversion of dpnp,
    # dpctl, or array_api inputs to numpy inputs would have occurred.

    lb, ub = 32768, 1048576  # lb is a patching condition, ub 2^20
    rand.seed(seed)
    shape = (rand.randint(lb, ub), 2)
    X = rand.uniform(high=np.finfo(dtype).max, size=shape).astype(dtype)
    sample_weight = rand.uniform(high=np.finfo(dtype).max, size=shape[0]).astype(dtype)

    if check:
        loc = rand.randint(0, shape[0] - 1)
        sample_weight[loc] = float(check)

    X = _convert_to_dataframe(
        X,
        target_df=dataframe,
        sycl_queue=queue,
    )
    sample_weight = _convert_to_dataframe(
        sample_weight,
        target_df=dataframe,
        sycl_queue=queue,
    )

    dispatch = {}
    if array_api_dispatch:
        if dataframe == "pandas":
            pytest.skip("pandas inputs do not work with sklearn's array_api_dispatch")
        dispatch["array_api_dispatch"] = array_api_dispatch

    with config_context(**dispatch):

        if check is None:
            X_out = _check_sample_weight(X, sample_weight)
            if dataframe == "pandas" or (
                dataframe == "array_api" and not array_api_dispatch
            ):
                assert isinstance(X, np.ndarray)
            else:
                assert type(X_out) == type(X)
        else:
            msg_err = "Input sample_weight contains NaN, infinity."
            with pytest.raises(ValueError, match=msg_err):
                X_out = _check_sample_weight(X, sample_weight)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "array_api_dispatch", [True, False] if sklearn_check_version("1.2") else [False]
)
@pytest.mark.parametrize(
    "dataframe, queue",
    get_dataframes_and_queues(
        "numpy,pandas" + ("dpctl,array_api" if sklearn_check_version("1.2") else "")
    ),
)
def test_validate_data_output(array_api_dispatch, dtype, dataframe, queue):
    # This testing assumes that array api inputs to validate_data will only occur
    # with sklearn array_api support which began in sklearn 1.2. This would assume
    # that somewhere upstream of the validate_data call, a data conversion of dpnp,
    # dpctl, or array_api inputs to numpy inputs would have occurred.
    est = DummyEstimator()
    X, y = gen_dataset(est, queue=queue, target_df=dataframe, dtype=dtype)[0]

    dispatch = {}
    if array_api_dispatch:
        if dataframe == "pandas":
            pytest.skip("pandas inputs do not work with sklearn's array_api_dispatch")
        dispatch["array_api_dispatch"] = array_api_dispatch

    with config_context(**dispatch):
        X_out, y_out = validate_data(est, X, y)
        # check sklearn validate_data operations work underneath
        X_array = validate_data(est, X, reset=False)

    if dataframe == "pandas" or (dataframe == "array_api" and not array_api_dispatch):
        # array_api_strict from sklearn < 1.2 and pandas will convert to numpy arrays
        assert isinstance(X_array, np.ndarray)
        assert isinstance(X_out, np.ndarray)
    else:
        assert type(X) == type(
            X_array
        ), f"validate_data converted {type(X)} to {type(X_array)}"
        assert type(X) == type(X_out), f"from_array converted {type(X)} to {type(X_out)}"
