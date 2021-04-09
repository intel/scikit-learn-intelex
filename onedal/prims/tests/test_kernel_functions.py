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
from onedal.prims import linear_kernel, rbf_kernel, poly_kernel
from sklearn.metrics.pairwise import rbf_kernel as sklearn_rbf_kernel


def test_dense_self_linear_kernel():
    rng = np.random.RandomState(0)
    X = np.array(5 * rng.random_sample((10, 4)))

    result = linear_kernel(X)
    expected = np.dot(X, np.array(X).T)
    assert_allclose(result, expected, rtol=1e-15)


def _test_dense_small_linear_kernel(scale, shift, dtype):
    rng = np.random.RandomState(0)
    X = np.array(5 * rng.random_sample((10, 4)), dtype=dtype)
    Y = np.array(5 * rng.random_sample((15, 4)), dtype=dtype)

    result = linear_kernel(X, Y, scale=scale, shift=shift)
    expected = np.dot(X, np.array(Y).T) * scale + shift
    tol = 1e-14 if dtype == np.float64 else 1e-6
    assert_allclose(result, expected, rtol=tol)


@pytest.mark.parametrize('scale', [1.0, 2.0])
@pytest.mark.parametrize('shift', [0.0, 1.0])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_dense_small_linear_kernel(scale, shift, dtype):
    _test_dense_small_linear_kernel(scale, shift, dtype)


def test_dense_self_rbf_kernel():
    rng = np.random.RandomState(0)
    X = np.array(5 * rng.random_sample((10, 4)))

    result = rbf_kernel(X)
    expected = sklearn_rbf_kernel(X)

    assert_allclose(result, expected, rtol=1e-14)


def _test_dense_small_rbf_kernel(gamma, dtype):
    rng = np.random.RandomState(0)
    X = np.array(5 * rng.random_sample((10, 4)), dtype=dtype)
    Y = np.array(5 * rng.random_sample((15, 4)), dtype=dtype)

    result = rbf_kernel(X, Y, gamma=gamma)
    expected = sklearn_rbf_kernel(X, Y, gamma)

    tol = 1e-14 if dtype == np.float64 else 1e-5
    assert_allclose(result, expected, rtol=tol)


@pytest.mark.parametrize('gamma', [0.1, None])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_dense_small_rbf_kernel(gamma, dtype):
    _test_dense_small_rbf_kernel(gamma, dtype)


def test_dense_self_poly_kernel():
    rng = np.random.RandomState(0)
    X = np.array(2 * rng.random_sample((10, 4)))

    degree = 2
    result = poly_kernel(X, degree=degree)
    expected = np.dot(X, np.array(X).T) ** degree

    assert_allclose(result, expected, rtol=1e-14)


def _test_dense_small_poly_kernel(gamma, coef0, degree, dtype):
    rng = np.random.RandomState(0)
    X = np.array(2 * rng.random_sample((10, 4)), dtype=dtype)
    Y = np.array(2 * rng.random_sample((15, 4)), dtype=dtype)

    result = poly_kernel(X, Y, gamma=gamma, coef0=coef0, degree=degree)
    expected = (gamma * np.dot(X, np.array(Y).T) + coef0) ** degree

    tol = 1e-14 if dtype == np.float64 else 1e-5
    assert_allclose(result, expected, rtol=tol)


@pytest.mark.parametrize('gamma', [0.1, 1.0])
@pytest.mark.parametrize('coef0', [0.0, 1.0])
@pytest.mark.parametrize('degree', [2, 3])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_dense_small_poly_kernel(gamma, coef0, degree, dtype):
    _test_dense_small_poly_kernel(gamma, coef0, degree, dtype)
