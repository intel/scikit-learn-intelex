#!/usr/bin/env python
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

import numpy as np
from numpy.testing import assert_allclose
from sklearn.datasets import make_regression
from daal4py.sklearn._utils import daal_check_version
from sklearn.datasets import make_classification
from scipy.special import expit
from sklearn._loss.loss import HalfBinomialLoss


def test_sklearnex_import_linear():
    from sklearnex.preview.objective_function import LinearModelLoss

    np.random.seed(42)
    X = np.random.normal(size=(100, 4))
    coef = np.array([1.0, -2.0, 3.1, -2.5])
    intercept = 2.8
    y = np.round(expit(X @ coef + intercept))

    coef2 = np.append(coef, intercept)

    onedal_loss = LinearModelLoss(base_loss=HalfBinomialLoss(), fit_intercept=True)

    logloss = onedal_loss.loss(coef2, X, y)

    gradient = onedal_loss.gradient(coef2, X, y)

    logloss2, gradient2 = onedal_loss.loss_gradient(coef2, X, y)

    gradient3, hessian, flag = onedal_loss.gradient_hessian(coef2, X, y)

    gradient4, hessp = onedal_loss.gradient_hessian_product(coef2, X, y)

    logloss_gth = 10.129922567126792

    gradient_gth = np.array(
        [1.52457206, 1.81682052, -1.06863193, 0.81881473, -1.34379483])

    hessian_gth = np.array([[5.18586773, 1.13656308, -1.32822742, 0.58710304, 0.37052724],
                            [1.13656308, 3.85940687, 0.38242507, -0.45595446, 0.70195371],
                            [-1.32822742, 0.38242507, 5.07365932, 1.62749146, -2.6480488],
                            [0.58710304, -0.45595446, 1.62749146, 5.95160586, 2.50535479],
                            [0.37052724, 0.70195371, -2.64804887, 2.50535479, 6.5056246]])

    hessp_res_gth = np.array(
        [5.95183367, 5.62439427, 3.10729957, 10.21560069, 7.43541147])

    assert_allclose(logloss, logloss_gth)
    assert_allclose(logloss2, logloss_gth)
    assert_allclose(gradient, gradient_gth)
    assert_allclose(gradient2, gradient_gth)
    assert_allclose(gradient3, gradient_gth)
    assert_allclose(gradient4, gradient_gth)
    assert_allclose(hessian, hessian_gth)
    assert_allclose(hessp(np.ones(5)), hessp_res_gth)

    if daal_check_version((2023, 'P', 200)):
        assert 'sklearnex' in onedal_loss.__module__
    else:
        assert 'sklearn' in onedal_loss.__module__
