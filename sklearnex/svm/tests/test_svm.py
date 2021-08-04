#!/usr/bin/env python
#===============================================================================
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
#===============================================================================

import numpy as np
from numpy.testing import assert_allclose

from dpctl import SyclQueue
from dpctl.tensor import usm_ndarray
from dpctl.memory import MemoryUSMDevice, MemoryUSMShared, MemoryUSMHost

def to_usm(queue, memtype, arrays):
    results = []
    for item in arrays:
        mem = memtype(item.nbytes, queue=queue)
        mem.copy_from_host(item.tobytes())
        results.append(usm_ndarray(item.shape, item.dtype, buffer=mem))
    return results


def test_sklearnex_usm_svc():
    """Tests if sklearn classifiers accept usm_ndarrays and return
    usm_ndarrays on the same device
    """
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearnex.svm import SVC

    q = SyclQueue('gpu')

    X, y = make_classification()
    Xtrain, Xtest, ytrain, ytest = to_usm(q, MemoryUSMDevice, train_test_split(X, y))

    estimator = SVC().fit(Xtrain, ytrain)
    ypred = estimator.predict(Xtest)

    assert hasattr(ypred, '__sycl_usm_array_interface__')
    assert ypred.sycl_queue.sycl_device == q.sycl_device
    assert ypred.shape == ytest.shape


def test_sklearnex_import_svc():
    from sklearnex.svm import SVC
    X = np.array([[-2, -1], [-1, -1], [-1, -2],
                  [+1, +1], [+1, +2], [+2, +1]])
    y = np.array([1, 1, 1, 2, 2, 2])
    svc = SVC(kernel='linear').fit(X, y)
    assert 'daal4py' in svc.__module__ or 'sklearnex' in svc.__module__
    assert_allclose(svc.dual_coef_, [[-0.25, .25]])
    assert_allclose(svc.support_, [1, 3])


def test_sklearnex_import_nusvc():
    from sklearnex.svm import NuSVC
    X = np.array([[-2, -1], [-1, -1], [-1, -2],
                  [+1, +1], [+1, +2], [+2, +1]])
    y = np.array([1, 1, 1, 2, 2, 2])
    svc = NuSVC(kernel='linear').fit(X, y)
    assert 'daal4py' in svc.__module__ or 'sklearnex' in svc.__module__
    assert_allclose(svc.dual_coef_, [[-0.04761905, -0.0952381, 0.0952381, 0.04761905]])
    assert_allclose(svc.support_, [0, 1, 3, 4])


def test_sklearnex_import_svr():
    from sklearnex.svm import SVR
    X = np.array([[-2, -1], [-1, -1], [-1, -2],
                  [+1, +1], [+1, +2], [+2, +1]])
    y = np.array([1, 1, 1, 2, 2, 2])
    svc = SVR(kernel='linear').fit(X, y)
    assert 'daal4py' in svc.__module__ or 'sklearnex' in svc.__module__
    assert_allclose(svc.dual_coef_, [[-0.1, 0.1]])
    assert_allclose(svc.support_, [1, 3])


# def test_sklearnex_import_nusvr():
#     from sklearnex.svm import NuSVR
#     X = np.array([[-2, -1], [-1, -1], [-1, -2],
#                   [+1, +1], [+1, +2], [+2, +1]])
#     y = np.array([1, 1, 1, 2, 2, 2])
#     svc = NuSVR(kernel='linear').fit(X, y)
#     assert 'daal4py' in svc.__module__ or 'sklearnex' in svc.__module__
#     assert_allclose(
#         svc.dual_coef_, [[0.55991593, -0.99563475,
#                           0.05571235, 0.88437172, -0.50436525]])
#     assert_allclose(svc.support_, [0, 1, 2, 3, 4])
