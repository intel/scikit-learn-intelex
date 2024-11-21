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
from sklearnex.utils.validation import validate_data


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

    X = _convert_to_dataframe(
        np.atleast_2d(X),
        target_df=dataframe,
        sycl_queue=queue,
    )  # test to see if convert_to_dataframe is causing problems
    X = np.atleast_2d(X)

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
        np.atleast_2d(X),
        target_df=dataframe,
        sycl_queue=queue,
    )  # test to see if convert_to_dataframe is causing problems
    X = np.atleast_2d(X)

    allow_nan = ensure_all_finite == "allow-nan"
    if check is None or (allow_nan and check == "NaN"):
        validate_data(est, X, ensure_all_finite=ensure_all_finite)
    else:
        type_err = "infinity" if allow_nan else "NaN, infinity"
        msg_err = f"Input X contains {type_err}."
        with pytest.raises(ValueError, match=msg_err):
            validate_data(est, X, ensure_all_finite=ensure_all_finite)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "array_api_dispatch", [True, False] if sklearn_check_version("1.2") else [False]
)
@pytest.mark.parametrize("dataframe, queue", get_dataframes_and_queues())
def test_validate_data_output(array_api_dispatch, dtype, dataframe, queue):
    est = DummyEstimator()
    X, y = gen_dataset(est, queue=queue, target_df=dataframe, dtype=dtype)[0]

    dispatch = {}
    if array_api_dispatch:
        pytest.skip(
            dataframe == "pandas",
            "pandas inputs do not work with sklearn's array_api_dispatch",
        )
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
