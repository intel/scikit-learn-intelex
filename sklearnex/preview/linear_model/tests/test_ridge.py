# ===============================================================================
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
# ===============================================================================

import numpy
import pytest
from numpy.testing import assert_allclose
from sklearn.exceptions import NotFittedError

from daal4py.sklearn._utils import daal_check_version
from onedal.tests.utils._dataframes_support import (
    _convert_to_dataframe,
    get_dataframes_and_queues,
)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import_ridge(dataframe, queue):
    from sklearnex.preview.linear_model import Ridge

    X = numpy.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = numpy.dot(X, numpy.array([1, 2])) + 3
    X_c = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y_c = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    ridge_reg = Ridge(alpha=0.5).fit(X_c, y_c)

    if daal_check_version((2024, "P", 600)):
        assert "preview" in ridge_reg.__module__
    else:
        assert "daal4py" in ridge_reg.__module__

    assert_allclose(ridge_reg.intercept_, 3.86, rtol=1e-2)
    assert_allclose(ridge_reg.coef_, [0.91, 1.64], rtol=1e-2)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("sample_size", [100, 1000])
@pytest.mark.parametrize("feature_size", [10, 50])
@pytest.mark.parametrize("alpha", [0.1, 0.5, 1.0])
def test_ridge_coefficients(dataframe, queue, sample_size, feature_size, alpha):
    from sklearnex.preview.linear_model import Ridge

    X = numpy.random.rand(sample_size, feature_size)
    y = numpy.random.rand(sample_size)
    X_c = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y_c = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    ridge_reg = Ridge(fit_intercept=False, alpha=alpha).fit(X_c, y_c)

    # computing the coefficients manually
    # using the normal equation formula: (X^T * X + lambda * I)^-1 * X^T * y
    lambda_identity = alpha * numpy.eye(X.shape[1])
    inverse_term = numpy.linalg.inv(numpy.dot(X.T, X) + lambda_identity)
    xt_y = numpy.dot(X.T, y)
    coefficients_manual = numpy.dot(inverse_term, xt_y)

    assert_allclose(ridge_reg.coef_, coefficients_manual, rtol=1e-6, atol=1e-6)


if daal_check_version((2024, "P", 600)):

    @pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
    def test_ridge_score_before_fit(dataframe, queue):
        from sklearnex.preview.linear_model import Ridge

        sample_count, feature_count = 10, 5

        model = Ridge(fit_intercept=True, alpha=0.5)

        X, y = numpy.random.rand(sample_count, feature_count), numpy.random.rand(
            sample_count
        )
        X_c = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
        y_c = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)

        with pytest.raises(NotFittedError):
            model.score(X_c, y_c)

    @pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
    def test_ridge_predict_before_fit(dataframe, queue):
        from sklearnex.preview.linear_model import Ridge

        sample_count, feature_count = 10, 5

        model = Ridge(fit_intercept=True, alpha=0.5)

        X = numpy.random.rand(sample_count, feature_count)
        X_c = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

        with pytest.raises(NotFittedError):
            model.predict(X_c)
