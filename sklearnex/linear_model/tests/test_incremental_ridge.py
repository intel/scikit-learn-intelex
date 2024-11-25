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
    from sklearnex.linear_model import IncrementalRidge

    def _compute_ridge_coefficients(X, y, alpha, fit_intercept):
        coefficients_manual, intercept_manual = None, None
        if fit_intercept:
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            X_centered = X - X_mean
            y_centered = y - y_mean

            X_with_intercept = np.hstack([np.ones((X.shape[0], 1)), X_centered])
            lambda_identity = alpha * np.eye(X_with_intercept.shape[1])
            inverse_term = np.linalg.inv(
                np.dot(X_with_intercept.T, X_with_intercept) + lambda_identity
            )
            xt_y = np.dot(X_with_intercept.T, y_centered)
            coefficients_manual = np.dot(inverse_term, xt_y)

            intercept_manual = y_mean - np.dot(X_mean, coefficients_manual[1:])
            coefficients_manual = coefficients_manual[1:]
        else:
            lambda_identity = alpha * np.eye(X.shape[1])
            inverse_term = np.linalg.inv(np.dot(X.T, X) + lambda_identity)
            xt_y = np.dot(X.T, y)
            coefficients_manual = np.dot(inverse_term, xt_y)

        return coefficients_manual, intercept_manual

    @pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
    @pytest.mark.parametrize("batch_size", [10, 100, 1000])
    @pytest.mark.parametrize("alpha", [0.1, 0.5, 1.0])
    @pytest.mark.parametrize("fit_intercept", [True, False])
    def test_inc_ridge_fit_coefficients(
        dataframe, queue, alpha, batch_size, fit_intercept
    ):
        sample_size, feature_size = 1000, 50
        X = np.random.rand(sample_size, feature_size)
        y = np.random.rand(sample_size)
        X_c = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
        y_c = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)

        inc_ridge = IncrementalRidge(
            fit_intercept=fit_intercept, alpha=alpha, batch_size=batch_size
        )
        inc_ridge.fit(X_c, y_c)

        coefficients_manual, intercept_manual = _compute_ridge_coefficients(
            X, y, alpha, fit_intercept
        )
        
        tol = 2e-4 if inc_ridge.coef_.dtype == np.float32 else 1e-6

        if fit_intercept:
            assert_allclose(inc_ridge.intercept_, intercept_manual, rtol=tol, atol=tol)

        assert_allclose(inc_ridge.coef_, coefficients_manual, rtol=tol, atol=tol)

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
        assert inc_ridge.score(X, y) >= 0.97

    @pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
    @pytest.mark.parametrize("fit_intercept", [True, False])
    def test_inc_ridge_predict_after_fit(dataframe, queue, fit_intercept):
        sample_size, feature_size = 1000, 50
        X = np.random.rand(sample_size, feature_size)
        y = np.random.rand(sample_size)
        X_c = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
        y_c = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)

        inc_ridge = IncrementalRidge(fit_intercept=fit_intercept, alpha=0.5)
        inc_ridge.fit(X_c, y_c)

        y_pred = inc_ridge.predict(X_c)

        coefficients_manual, intercept_manual = _compute_ridge_coefficients(
            X, y, 0.5, fit_intercept
        )
        y_pred_manual = np.dot(X, coefficients_manual)
        if fit_intercept:
            y_pred_manual += intercept_manual
        
        tol = 1e-5 if inc_ridge.coef_.dtype == np.float32 else 1e-6

        assert_allclose(_as_numpy(y_pred), y_pred_manual, rtol=tol, atol=tol)
