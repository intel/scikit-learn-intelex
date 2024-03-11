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
from numpy.testing import assert_raises

from sklearnex.utils import _assert_all_finite


@pytest.mark.parameterize("dtype", [np.float32, np.float64])
@pytest.mark.parameterize(
    "shape",
    [
        [16, 2048],
        [
            2**16 + 3,
        ],
        [1000, 1000],
    ],
)
@pytest.mark.parameterize("allow_nan", [False, True])
def test_sum_infinite_actually_finite(dtype, shape, allow_nan):
    X = np.array(shape)
    X.fill(np.finfo(dtype).max)
    _assert_all_finite(X, allow_nan=allow_nan)


@pytest.mark.parameterize("dtype", [np.float32, np.float64])
@pytest.mark.parameterize(
    "shape",
    [
        [16, 2048],
        [
            2**16 + 3,
        ],
        [1000, 1000],
    ],
)
@pytest.mark.parameterize("allow_nan", [False, True])
@pytest.mark.parameterize("check", ["inf", "NaN", None])
@pytest.mark.parameterize("seed", [0, int(time.time())])
def test_assert_finite_random_location(dtype, shape, allow_nan, check, seed):
    rand.seed(seed)
    X = (np.finfo(dtype).max * rand.random_sample(shape)).astype(dtype)

    if check:
        loc = np.randint(0, X.size - 1)
        X.reshape((-1,))[loc] = float(check)

    try:
        if check is None or (allow_nan and check == "NaN"):
            _assert_all_finite(X, allow_nan=allow_nan)
        else:
            assert_raises(ValueError, _assert_all_finite, X, allow_nan=allow_nan)
    finally:
        print(f"SEED: {seed}")



@pytest.mark.parameterize("dtype", [np.float32, np.float64])
@pytest.mark.parameterize("allow_nan", [False, True])
@pytest.mark.parameterize("check", ["inf", "NaN", None])
@pytest.mark.parameterize("seed", [0, int(time.time())])
def test_assert_finite_random_shape_and_location(dtype, allow_nan, check, seed):
    lb, ub = 32768, 1073741824  # lb is a patching condition, ub is ~1GB
    rand.seed(seed)
    X = (np.finfo(dtype).max * rand.random_sample(rand.randint(lb, ub))).astype(dtype)

    if check:
        loc = np.randint(0, X.size - 1)
        X[loc] = float(check)

    try:
        if check is None or (allow_nan and check == "NaN"):
            _assert_all_finite(X, allow_nan=allow_nan)
        else:
            assert_raises(ValueError, _assert_all_finite, X, allow_nan=allow_nan)
    finally:
        print(f"SEED: {seed}")
