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

from daal4py.sklearn._utils import daal_check_version, sklearn_check_version
from sklearn.metrics import log_loss

if daal_check_version((2023, 'P', 100)):
    import pytest
    import numpy as np
    from numpy.testing import assert_allclose

    from onedal.objective_function import LogisticLoss
    from onedal.tests.utils._device_selection import get_queues

    @pytest.mark.parametrize('queue', get_queues())
    @pytest.mark.parametrize('dtype', [np.float32, np.float64])
    def test_logloss(queue, dtype):
        seed = 42
        row_count, col_count = 1000, 29

        gen = np.random.default_rng(seed)
        data = gen.uniform(low=-0.5, high=+0.6,
                           size=(row_count, col_count))
        coef = gen.uniform(low =-0.5, high=+0.6, size=(col_count + 1, 1)).astype(dtype=dtype)
        y_true = gen.integers(0, 2, (row_count, 1))
        data = data.astype(dtype=dtype)

        alg = LogisticLoss(queue=queue)
        logloss_onedal = alg.loss(coef, data, y_true, fit_intercept=True)
        logloss_onedal_no_intercept = alg.loss(coef, data, y_true, fit_intercept=False)

        def bind_probs(x):
            eps = 1e-7 if dtype == np.float32 else 1e-15
            return 0.0 if x < eps else 1.0 if x > (1 - eps) else x
        probs = 1 / (1 + np.exp(-((data @ coef[1:]) + coef[0])))
        probs = np.array(list(map(bind_probs, probs)))
        probs_no_intercept = 1 / (1 + np.exp(-((data @ coef[1:]))))
        probs_no_intercept = np.array(list(map(bind_probs, probs_no_intercept)))


        logloss_gth = log_loss(y_true, probs)
        logloss_gth_no_intercept = log_loss(y_true, probs_no_intercept)

        tol = 1e-5 if dtype == np.float32 else 1e-7
        assert_allclose(logloss_onedal, logloss_gth, rtol=tol)
        assert_allclose(logloss_onedal_no_intercept, logloss_gth_no_intercept, rtol=tol)