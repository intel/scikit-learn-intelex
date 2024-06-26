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
    from sklearn.linear_model import Ridge as Ridge_sklearn

    from daal4py.sklearn.linear_model._ridge import Ridge

    X, y = make_regression(n_samples=10, n_features=5, n_targets=3, random_state=0)

    # asserting exception if alpha has wrong shape
    wrong_alpha_shape = numpy.random.rand(5)
    with pytest.raises(ValueError):
        model_daal = Ridge(alpha=wrong_alpha_shape).fit(X, y)

    alpha = 3 + numpy.random.rand(3) * 5
    model_daal = Ridge(alpha=alpha)
    model_sklearn = Ridge_sklearn(alpha=alpha)

    model_daal.fit(X, y)
    model_sklearn.fit(X, y)

    numpy.testing.assert_allclose(model_daal.coef_, model_sklearn.coef_, rtol=1e-5)
