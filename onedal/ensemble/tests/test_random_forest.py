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
import numpy as np
from numpy.testing import assert_allclose

from onedal.ensemble import RandomForestClassifier, RandomForestRegressor
from onedal.tests.utils._device_selection import get_queues

from sklearn.datasets import make_classification, make_regression

# TODO:
# will be replaced with common check.
try:
    import dpctl
    import dpctl.tensor as dpt
    dpctl_available = True
except ImportError:
    dpctl_available = False


@pytest.mark.parametrize('queue', get_queues())
def test_rf_classifier(queue):
    X, y = make_classification(n_samples=100, n_features=4,
                               n_informative=2, n_redundant=0,
                               random_state=0, shuffle=False)
    rf = RandomForestClassifier(
        max_depth=2, random_state=0).fit(X, y, queue=queue)
    assert_allclose([1], rf.predict([[0, 0, 0, 0]], queue=queue))


@pytest.mark.skipif(not dpctl_available,
                    reason="requires dpctl")
@pytest.mark.parametrize('queue', get_queues())
def test_rf_classifier_dpctl(queue):
    queue = dpctl.SyclQueue('gpu')
    X, y = make_classification(n_samples=100, n_features=4,
                               n_informative=2, n_redundant=0,
                               random_state=0, shuffle=False)
    dpt_X = dpt.asarray(X, usm_type="device", sycl_queue=queue)
    dpt_y = dpt.asarray(y, usm_type="device", sycl_queue=queue)
    rf = RandomForestClassifier(
        max_depth=2, random_state=0).fit(dpt_X, dpt_y)
    dpt_X_test = dpt.asarray([[0, 0, 0, 0]], usm_type="device", sycl_queue=queue)
    # For assert_allclose check
    # copy dpctl tensor data to host.
    assert_allclose([1], dpt.to_numpy(rf.predict(dpt_X_test)))


@pytest.mark.skipif(not dpctl_available,
                    reason="requires dpctl")
@pytest.mark.parametrize('queue', get_queues())
def test_rf_classifier_dpctl_w_explicit_queue(queue):
    queue = dpctl.SyclQueue('gpu')
    X, y = make_classification(n_samples=100, n_features=4,
                               n_informative=2, n_redundant=0,
                               random_state=0, shuffle=False)
    dpt_X = dpt.asarray(X, usm_type="device", sycl_queue=queue)
    dpt_y = dpt.asarray(y, usm_type="device", sycl_queue=queue)
    rf = RandomForestClassifier(
        max_depth=2, random_state=0).fit(dpt_X, dpt_y, queue=queue)
    dpt_X_test = dpt.asarray([[0, 0, 0, 0]], usm_type="device", sycl_queue=queue)
    # For assert_allclose check
    # copy dpctl tensor data to host.
    assert_allclose([1], dpt.to_numpy(rf.predict(dpt_X_test, queue=queue)))


@pytest.mark.parametrize('queue', get_queues())
def test_rf_regression(queue):
    X, y = make_regression(n_samples=100, n_features=4, n_informative=2,
                           random_state=0, shuffle=False)
    rf = RandomForestRegressor(
        max_depth=2, random_state=0).fit(X, y, queue=queue)
    assert_allclose(
        [-6.83], rf.predict([[0, 0, 0, 0]], queue=queue), atol=1e-2)
