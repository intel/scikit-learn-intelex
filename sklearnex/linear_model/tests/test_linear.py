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
import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import make_regression

from daal4py.sklearn._utils import daal_check_version
from daal4py.sklearn.linear_model.tests.test_ridge import (
    _test_multivariate_ridge_alpha_shape,
    _test_multivariate_ridge_coefficients,
)
from onedal.tests.utils._dataframes_support import (
    _as_numpy,
    _convert_to_dataframe,
    get_dataframes_and_queues,
)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("macro_block", [None, 1024])
def test_sklearnex_import_linear(dataframe, queue, dtype, macro_block):
    from sklearnex.linear_model import LinearRegression

    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    X = X.astype(dtype=dtype)
    y = y.astype(dtype=dtype)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)

    linreg = LinearRegression()
    if daal_check_version((2024, "P", 0)) and macro_block is not None:
        hparams = linreg.get_hyperparameters("fit")
        hparams.cpu_macro_block = macro_block
        hparams.gpu_macro_block = macro_block

    linreg.fit(X, y)

    assert hasattr(linreg, "_onedal_estimator")
    assert "sklearnex" in linreg.__module__
    assert linreg.n_features_in_ == 2

    tol = 1e-5 if _as_numpy(linreg.coef_).dtype == np.float32 else 1e-7
    assert_allclose(_as_numpy(linreg.intercept_), 3.0, rtol=tol)
    assert_allclose(_as_numpy(linreg.coef_), [1.0, 2.0], rtol=tol)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import_ridge(dataframe, queue):
    from sklearnex.linear_model import Ridge

    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    ridgereg = Ridge().fit(X, y)
    assert "daal4py" in ridgereg.__module__
    assert_allclose(ridgereg.intercept_, 4.5)
    assert_allclose(ridgereg.coef_, [0.8, 1.4])


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import_lasso(dataframe, queue):
    from sklearnex.linear_model import Lasso

    X = [[0, 0], [1, 1], [2, 2]]
    y = [0, 1, 2]
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    lasso = Lasso(alpha=0.1).fit(X, y)
    assert "daal4py" in lasso.__module__
    assert_allclose(lasso.intercept_, 0.15)
    assert_allclose(lasso.coef_, [0.85, 0.0])


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import_elastic(dataframe, queue):
    from sklearnex.linear_model import ElasticNet

    X, y = make_regression(n_features=2, random_state=0)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    elasticnet = ElasticNet(random_state=0).fit(X, y)
    assert "daal4py" in elasticnet.__module__
    assert_allclose(elasticnet.intercept_, 1.451, atol=1e-3)
    assert_allclose(elasticnet.coef_, [18.838, 64.559], atol=1e-3)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sklearnex_reconstruct_model(dataframe, queue, dtype):
    from sklearnex.linear_model import LinearRegression

    seed = 42
    num_samples = 3500
    num_features, num_targets = 14, 9

    gen = np.random.default_rng(seed)
    intercept = gen.random(size=num_targets, dtype=dtype)
    coef = gen.random(size=(num_targets, num_features), dtype=dtype).T

    X = gen.random(size=(num_samples, num_features), dtype=dtype)
    gtr = X @ coef + intercept[np.newaxis, :]

    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    linreg = LinearRegression(fit_intercept=True)
    linreg.coef_ = coef.T
    linreg.intercept_ = intercept

    y_pred = linreg.predict(X)

    tol = 1e-5 if _as_numpy(y_pred).dtype == np.float32 else 1e-7
    assert_allclose(gtr, _as_numpy(y_pred), rtol=tol)


def test_sklearnex_multivariate_ridge_coefs():
    from sklearnex.linear_model import Ridge

    _test_multivariate_ridge_coefficients(Ridge, random_state=0)


def test_sklearnex_multivariate_ridge_alpha_shape():
    from sklearnex.linear_model import Ridge

    _test_multivariate_ridge_alpha_shape(Ridge, random_state=0)
