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

import pytest
import numpy as np
from numpy.testing import assert_allclose
from onedal.tests.utils._device_selection import get_queues


if daal_check_version((2023, 'P', 200)) and sklearn_check_version('1.1'):
    from sklearn._loss.loss import HalfBinomialLoss
    from sklearn.linear_model import _linear_loss
    from onedal.objective_function import LogisticLoss

    def gen_test(
        row_count=5000, col_count=29, seed=42, dtype=np.float32, fit_intercept=True
    ):
        gen = np.random.default_rng(seed)
        data = gen.uniform(low=-0.5, high=+0.6, size=(row_count, col_count))
        coef = gen.uniform(low=-0.5, high=+0.6, size=(col_count + 1, 1)) \
                  .astype(dtype=dtype) \
                  .reshape(-1)
        y_true = gen.integers(0, 2, (row_count, 1)).astype(
            dtype=dtype).reshape(-1)
        data = data.astype(dtype=dtype)

        if not fit_intercept:
            coef = coef[:-1]
        return coef, data, y_true

    @pytest.mark.parametrize("queue", get_queues())
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("fit_intercept", [True, False])
    @pytest.mark.parametrize("l2_reg_strength", [0.0, 1.7])
    @pytest.mark.parametrize("seed", [16, 42, 2007])
    def test_logloss(queue, dtype, fit_intercept, l2_reg_strength, seed):
        tol = 1e-4 if dtype == np.float32 else 1e-7

        coef, data, y_true = gen_test(dtype=dtype, fit_intercept=fit_intercept, seed=seed)

        alg = LogisticLoss(fit_intercept=fit_intercept)
        logloss_onedal = alg.loss(
            coef, data, y_true, l2_reg_strength=l2_reg_strength, queue=queue)

        alg_internal = _linear_loss.LinearModelLoss(
            base_loss=HalfBinomialLoss(), fit_intercept=fit_intercept,
        )

        logloss_gth = alg_internal.loss(
            coef, data, y_true, l2_reg_strength=l2_reg_strength
        )

        assert_allclose(logloss_onedal, logloss_gth, rtol=tol)

    @pytest.mark.parametrize("queue", get_queues())
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("fit_intercept", [True, False])
    @pytest.mark.parametrize("l2_reg_strength", [0.0, 1.7])
    @pytest.mark.parametrize("seed", [16, 42, 2007])
    def test_gradient(queue, dtype, fit_intercept, l2_reg_strength, seed):
        tol = 1e-4 if dtype == np.float32 else 1e-7

        coef, data, y_true = gen_test(dtype=dtype, fit_intercept=fit_intercept, seed=seed)

        alg = LogisticLoss(fit_intercept=fit_intercept)
        gradient_onedal = alg.gradient(
            coef, data, y_true, l2_reg_strength=l2_reg_strength, queue=queue
        )

        alg_internal = _linear_loss.LinearModelLoss(
            base_loss=HalfBinomialLoss(), fit_intercept=fit_intercept
        )
        gradient_gth = alg_internal.gradient(
            coef, data, y_true, l2_reg_strength=l2_reg_strength
        )

        assert_allclose(gradient_onedal, gradient_gth, rtol=tol)

    @pytest.mark.parametrize("queue", get_queues())
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("fit_intercept", [True, False])
    @pytest.mark.parametrize("l2_reg_strength", [0.0, 1.7])
    @pytest.mark.parametrize("seed", [16, 42, 2007])
    def test_loss_gradient(queue, dtype, fit_intercept, l2_reg_strength, seed):
        tol = 1e-4 if dtype == np.float32 else 1e-7

        coef, data, y_true = gen_test(dtype=dtype, fit_intercept=fit_intercept, seed=seed)

        alg = LogisticLoss(fit_intercept=fit_intercept)
        logloss_onedal, gradient_onedal = alg.loss_gradient(
            coef, data, y_true, l2_reg_strength=l2_reg_strength, queue=queue
        )

        alg_internal = _linear_loss.LinearModelLoss(
            base_loss=HalfBinomialLoss(), fit_intercept=fit_intercept
        )

        logloss_gth, gradient_gth = alg_internal.loss_gradient(
            coef, data, y_true, l2_reg_strength=l2_reg_strength
        )

        assert_allclose(logloss_onedal, logloss_gth, rtol=tol)
        assert_allclose(gradient_onedal, gradient_gth, rtol=tol)

    @pytest.mark.parametrize("queue", get_queues())
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("fit_intercept", [True, False])
    @pytest.mark.parametrize("l2_reg_strength", [0.0, 1.7])
    @pytest.mark.parametrize("seed", [16, 42, 2007])
    def test_gradient_hessian(queue, dtype, fit_intercept, l2_reg_strength, seed):
        tol = 1e-3 if dtype == np.float32 else 1e-7

        coef, data, y_true = gen_test(dtype=dtype, fit_intercept=fit_intercept, seed=seed)

        alg = LogisticLoss(fit_intercept=fit_intercept)
        gradient_onedal, hessian_onedal, status_onedal = alg.gradient_hessian(
            coef, data, y_true, l2_reg_strength=l2_reg_strength, queue=queue
        )

        alg_internal = _linear_loss.LinearModelLoss(
            base_loss=HalfBinomialLoss(), fit_intercept=fit_intercept
        )

        gradient_gth, hessian_gth, status_gth = alg_internal.gradient_hessian(
            coef, data, y_true, l2_reg_strength=l2_reg_strength
        )

        assert_allclose(gradient_onedal, gradient_gth, rtol=tol)
        assert_allclose(hessian_onedal, hessian_gth, rtol=tol)
        assert status_onedal == status_gth

    @pytest.mark.parametrize("queue", get_queues())
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("fit_intercept", [True, False])
    @pytest.mark.parametrize("l2_reg_strength", [0.0, 1.7])
    @pytest.mark.parametrize("seed", [16, 42, 2007])
    def test_gradient_hessian_product(queue, dtype, fit_intercept, l2_reg_strength, seed):
        tol = 1e-3 if dtype == np.float32 else 1e-7

        coef, data, y_true = gen_test(dtype=dtype, fit_intercept=fit_intercept, seed=seed)

        alg = LogisticLoss(fit_intercept=fit_intercept)
        gradient_onedal, hessp_onedal = alg.gradient_hessian_product(
            coef, data, y_true, l2_reg_strength=l2_reg_strength, queue=queue
        )

        alg_internal = _linear_loss.LinearModelLoss(
            base_loss=HalfBinomialLoss(), fit_intercept=fit_intercept
        )
        gradient_gth, hessp_gth = alg_internal.gradient_hessian_product(
            coef, data, y_true, l2_reg_strength=l2_reg_strength
        )

        assert_allclose(gradient_onedal, gradient_gth, rtol=tol)

        size = gradient_gth.shape[0]

        gen = np.random.default_rng(42)

        for i in range(5):
            s = gen.normal(size=size)

            product_onedal = hessp_onedal(s)
            product_gth = hessp_gth(s)

            assert_allclose(product_onedal, product_gth, rtol=tol)

    @pytest.mark.parametrize("queue", get_queues())
    @pytest.mark.parametrize(
        "dimensions", [(100000, 15), (1000, 50), (100, 100), (500, 20), (20, 73)]
    )
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_sizes(queue, dimensions, dtype):
        fit_intercept = True
        l2_reg_strength = 3.1

        row_count = dimensions[0]
        col_count = dimensions[1]
        coef, data, y_true = gen_test(
            row_count=row_count,
            col_count=col_count,
            dtype=dtype,
            fit_intercept=fit_intercept,
        )
        alg = LogisticLoss(fit_intercept=fit_intercept)
        logloss_onedal, gradient_onedal = alg.loss_gradient(
            coef, data, y_true, l2_reg_strength=l2_reg_strength, queue=queue
        )

        alg_internal = _linear_loss.LinearModelLoss(
            base_loss=HalfBinomialLoss(), fit_intercept=fit_intercept
        )

        logloss_gth, gradient_gth = alg_internal.loss_gradient(
            coef, data, y_true, l2_reg_strength=l2_reg_strength
        )

        assert_allclose(logloss_onedal, logloss_gth, rtol=1e-7)
        assert_allclose(gradient_onedal, gradient_gth, rtol=1e-7)
