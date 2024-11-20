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
from sklearnex.utils import validate_data


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
@pytest.mark.parametrize("allow_nan", [False, True])
def test_sum_infinite_actually_finite(dtype, shape, allow_nan):
    est = DummyEstimator()
    X = np.array(shape, dtype=dtype)
    X.fill(np.finfo(dtype).max)
    X_array = validate_data(est, X, allow_nan=allow_nan)
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
@pytest.mark.parametrize("allow_nan", [False, True])
@pytest.mark.parametrize("check", ["inf", "NaN", None])
@pytest.mark.parametrize("seed", [0, int(time.time())])
@pytest.mark.parametrize("dataframe, queue", get_dataframes_and_queues())
def test_validate_data_random_location(
    dataframe, queue, dtype, shape, allow_nan, check, seed
):
    est = DummyEstimator()
    rand.seed(seed)
    X = rand.uniform(high=np.finfo(dtype).max, size=shape).astype(dtype)

    if check:
        loc = rand.randint(0, X.size - 1)
        X.reshape((-1,))[loc] = float(check)

    X = _convert_to_dataframe(
        X,
        target_df=dataframe,
        sycl_queue=queue,
    )

    if check is None or (allow_nan and check == "NaN"):
        validate_data(est, X, allow_nan=allow_nan)
    else:
        msg_err = "Input contains " + ("infinity" if allow_nan else "NaN, infinity") + "."
        with pytest.raises(ValueError, match=msg_err):
            validate_data(est, X, allow_nan=allow_nan)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("allow_nan", [False, True])
@pytest.mark.parametrize("check", ["inf", "NaN", None])
@pytest.mark.parametrize("seed", [0, int(time.time())])
@pytest.mark.parametrize("dataframe, queue", get_dataframes_and_queues())
def test_validate_data_random_shape_and_location(
    dataframe, queue, dtype, allow_nan, check, seed
):
    est = DummyEstimator()
    lb, ub = 32768, 1048576  # lb is a patching condition, ub 2^20
    rand.seed(seed)
    X = rand.uniform(high=np.finfo(dtype).max, size=rand.randint(lb, ub)).astype(dtype)

    if check:
        loc = rand.randint(0, X.size - 1)
        X[loc] = float(check)

    X = _convert_to_dataframe(
        X,
        target_df=dataframe,
        sycl_queue=queue,
    )

    if check is None or (allow_nan and check == "NaN"):
        validate_data(est, X)
    else:
        msg_err = "Input contains " + ("infinity" if allow_nan else "NaN, infinity") + "."
        with pytest.raises(ValueError, match=msg_err):
            validate_data(est, X, allow_nan=allow_nan)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("array_api_dispatch", [True, False])
@pytest.mark.parametrize(
    "dataframe, queue", get_dataframes_and_queues("numpy,dpctl,dpnp")
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_validate_data_output(array_api_dispatch, dtype, dataframe, queue):
    est = DummyEstimator()
    X, y = gen_dataset(est, queue=queue, target_df=dataframe, dtype=dtype)[0]

    dispatch = {}
    if sklearn_check_version("1.2"):
        dispatch["array_api_dispatch"] = array_api_dispatch

    with config_context(**dispatch):
        validate_data(est, X, y)
        est.fit(X, y)
        X_array = validate_data(est, X, reset=False)
        X_out = est.predict(X)

    if dataframe == "pandas" or (
        dataframe == "array_api"
        and not (sklearn_check_version("1.2") and array_api_dispatch)
    ):
        # array_api_strict from sklearn < 1.2 and pandas will convert to numpy arrays
        assert isinstance(X_array, np.ndarray)
        assert isinstance(X_out, np.ndarray)
    else:
        assert type(X) == type(
            X_array
        ), f"validate_data converted {type(X)} to {type(X_array)}"
        assert type(X) == type(X_out), f"from_array converted {type(X)} to {type(X_out)}"
