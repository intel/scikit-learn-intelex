# ===============================================================================
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
# ===============================================================================

import pytest
import numpy as np
from numpy.testing import assert_allclose
from sklearn import datasets, metrics
from onedal.prims import linear_kernel

from sklearn.utils.estimator_checks import check_estimator
import sklearn.utils.estimator_checks


def _test_input_format_c_contiguous_numpy(dtype):
    rng = np.random.RandomState(0)
    x_default = np.array(5 * rng.random_sample((10, 4)), dtype=dtype)

    x_numpy = np.asanyarray(x_default, dtype=dtype, order='C')
    assert x_numpy.flags.c_contiguous
    assert not x_numpy.flags.f_contiguous
    assert not x_numpy.flags.fnc

    expected = linear_kernel(x_default)
    result = linear_kernel(x_numpy)
    assert_allclose(expected, result)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_input_format_c_contiguous_numpy(dtype):
    _test_input_format_c_contiguous_numpy(dtype)


def _test_input_format_f_contiguous_numpy(dtype):
    rng = np.random.RandomState(0)
    x_default = np.array(5 * rng.random_sample((10, 4)), dtype=dtype)

    x_numpy = np.asanyarray(x_default, dtype=dtype, order='F')
    assert not x_numpy.flags.c_contiguous
    assert x_numpy.flags.f_contiguous
    assert x_numpy.flags.fnc

    expected = linear_kernel(x_default)
    result = linear_kernel(x_numpy)
    assert_allclose(expected, result)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_input_format_f_contiguous_numpy(dtype):
    _test_input_format_f_contiguous_numpy(dtype)


def _test_input_format_c_not_contiguous_numpy(dtype):
    rng = np.random.RandomState(0)
    x_default = np.array(5 * rng.random_sample((10, 4)), dtype=dtype)

    dummy_data = np.insert(x_default, range(1, x_default.shape[1]), 8, axis=1)
    x_numpy = dummy_data[:, ::2]

    assert_allclose(x_numpy, x_default)

    assert not x_numpy.flags.c_contiguous
    assert not x_numpy.flags.f_contiguous
    assert not x_numpy.flags.fnc

    expected = linear_kernel(x_default)
    result = linear_kernel(x_numpy)
    assert_allclose(expected, result)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_input_format_c_not_contiguous_numpy(dtype):
    _test_input_format_c_not_contiguous_numpy(dtype)


def _test_input_format_c_contiguous_pandas(dtype):
    pd = pytest.importorskip('pandas')
    rng = np.random.RandomState(0)
    x_default = np.array(5 * rng.random_sample((10, 4)), dtype=dtype)

    x_numpy = np.asanyarray(x_default, dtype=dtype, order='C')
    assert x_numpy.flags.c_contiguous
    assert not x_numpy.flags.f_contiguous
    assert not x_numpy.flags.fnc
    x_df = pd.DataFrame(x_numpy)

    expected = linear_kernel(x_df)
    result = linear_kernel(x_numpy)
    assert_allclose(expected, result)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_input_format_c_contiguous_pandas(dtype):
    _test_input_format_c_contiguous_pandas(dtype)


def _test_input_format_f_contiguous_pandas(dtype):
    pd = pytest.importorskip('pandas')
    rng = np.random.RandomState(0)
    x_default = np.array(5 * rng.random_sample((10, 4)), dtype=dtype)

    x_numpy = np.asanyarray(x_default, dtype=dtype, order='F')
    assert not x_numpy.flags.c_contiguous
    assert x_numpy.flags.f_contiguous
    assert x_numpy.flags.fnc
    x_df = pd.DataFrame(x_numpy)

    expected = linear_kernel(x_df)
    result = linear_kernel(x_numpy)
    assert_allclose(expected, result)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_input_format_f_contiguous_pandas(dtype):
    _test_input_format_f_contiguous_pandas(dtype)
