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
import numpy
import pytest
import sklearn
from sklearn.datasets import make_regression


def test_multivariate_ridge_coefficients():
    from daal4py.sklearn.linear_model._ridge import Ridge

    X, y = make_regression(n_samples=10, n_features=5, n_targets=3, random_state=0)

    # asserting exception if alpha has wrong shape
    wrong_alpha_shape = numpy.random.rand(5)
    with pytest.raises(ValueError):
        model = Ridge(alpha=wrong_alpha_shape).fit(X, y)

    # computing coefficients using daal4py Ridge
    alpha = 3 + numpy.random.rand(3) * 5
    model = Ridge(fit_intercept=False, alpha=alpha)

    model.fit(X, y)

    # computing coefficients manually
    n_features, n_targets = X.shape[1], y.shape[1]
    betas = numpy.zeros((n_targets, n_features))

    identity_matrix = numpy.eye(n_features)

    for j in range(n_targets):
        y_j = y[:, j]
        inverse_term = numpy.linalg.inv(numpy.dot(X.T, X) + alpha[j] * identity_matrix)
        beta_j = numpy.dot(inverse_term, numpy.dot(X.T, y_j))
        betas[j, :] = beta_j

    # asserting that the coefficients are close
    numpy.testing.assert_allclose(model.coef_, betas, rtol=1e-3, atol=1e-3)
