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

from daal4py.sklearn._utils import daal_check_version

if daal_check_version((2024, "P", 600)):
    import numpy as np
    import pytest
    from numpy.testing import assert_allclose
    from sklearn.exceptions import NotFittedError

    from onedal.tests.utils._dataframes_support import (
        _as_numpy,
        _convert_to_dataframe,
        get_dataframes_and_queues,
    )
    from sklearnex.preview.linear_model import IncrementalRidge

    @pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
    @pytest.mark.parametrize("batch_size", [10, 100, 1000])
    @pytest.mark.parametrize("alpha", [0.1, 0.5, 1.0])
    def test_inc_ridge_fit_coefficients(dataframe, queue, alpha, batch_size):
        sample_size, feature_size = 1000, 50
        X = np.random.rand(sample_size, feature_size)
        y = np.random.rand(sample_size)
        X_c = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
        y_c = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)

        inc_ridge = IncrementalRidge(
            fit_intercept=False, alpha=alpha, batch_size=batch_size
        )
        inc_ridge.fit(X_c, y_c)

        lambda_identity = alpha * np.eye(X.shape[1])
        inverse_term = np.linalg.inv(np.dot(X.T, X) + lambda_identity)
        xt_y = np.dot(X.T, y)
        coefficients_manual = np.dot(inverse_term, xt_y)

        assert_allclose(inc_ridge.coef_, coefficients_manual, rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
    @pytest.mark.parametrize("batch_size", [2, 5])
    @pytest.mark.parametrize("alpha", [0.1, 0.5, 1.0])
    def test_inc_ridge_partial_fit_coefficients(dataframe, queue, alpha, batch_size):
        sample_size, feature_size = 1000, 50
        X = np.random.rand(sample_size, feature_size)
        y = np.random.rand(sample_size)
        X_split = np.array_split(X, batch_size)
        y_split = np.array_split(y, batch_size)

        inc_ridge = IncrementalRidge(fit_intercept=False, alpha=alpha)

        for batch_index in range(len(X_split)):
            X_c = _convert_to_dataframe(
                X_split[batch_index], sycl_queue=queue, target_df=dataframe
            )
            y_c = _convert_to_dataframe(
                y_split[batch_index], sycl_queue=queue, target_df=dataframe
            )
            inc_ridge.partial_fit(X_c, y_c)

        lambda_identity = alpha * np.eye(X.shape[1])
        inverse_term = np.linalg.inv(np.dot(X.T, X) + lambda_identity)
        xt_y = np.dot(X.T, y)
        coefficients_manual = np.dot(inverse_term, xt_y)

        assert_allclose(inc_ridge.coef_, coefficients_manual, rtol=1e-6, atol=1e-6)

    def test_inc_ridge_score_before_fit():
        X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        y = np.dot(X, np.array([1, 2])) + 3
        inc_ridge = IncrementalRidge(alpha=0.5)
        with pytest.raises(NotFittedError):
            inc_ridge.score(X, y)

    def test_inc_ridge_predict_before_fit():
        X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        inc_ridge = IncrementalRidge(alpha=0.5)
        with pytest.raises(NotFittedError):
            inc_ridge.predict(X)

    def test_inc_ridge_score_after_fit():
        X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        y = np.dot(X, np.array([1, 2])) + 3
        inc_ridge = IncrementalRidge(alpha=0.5)
        inc_ridge.fit(X, y)
        assert inc_ridge.score(X, y) > 0.0

    def test_inc_ridge_predict_after_fit():
        X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        y = np.dot(X, np.array([1, 2])) + 3
        inc_ridge = IncrementalRidge(alpha=0.5)
        inc_ridge.fit(X, y)
        assert inc_ridge.predict(X).shape[0] == y.shape[0]
