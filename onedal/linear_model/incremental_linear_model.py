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

import numpy as np

from daal4py.sklearn._utils import get_dtype

from ..common.hyperparameters import get_hyperparameters
from ..datatypes import _convert_to_supported, from_table, to_table
from ..utils import _check_X_y, _num_features
from .linear_model import BaseLinearRegression


class IncrementalLinearRegression(BaseLinearRegression):
    """
    Incremental Linear Regression oneDAL implementation.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    algorithm : string, default="norm_eq"
        Algorithm used for computation on oneDAL side
    """

    def __init__(self, fit_intercept=True, copy_X=False, algorithm="norm_eq"):
        module = self._get_backend("linear_model", "regression")
        super().__init__(fit_intercept=fit_intercept, copy_X=copy_X, algorithm=algorithm)
        self._partial_result = module.partial_train_result()

    def _reset(self):
        module = self._get_backend("linear_model", "regression")
        self._partial_result = module.partial_train_result()

    def partial_fit(self, X, y, queue=None):
        """
        Computes partial data for linear regression
        from data batch X and saves it to `_partial_result`.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data batch, where `n_samples` is the number of samples
            in the batch, and `n_features` is the number of features.

        y: array-like of shape (n_samples,) or (n_samples, n_targets) in
            case of multiple targets
            Responses for training data.

        queue : dpctl.SyclQueue
            If not None, use this queue for computations.
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        module = self._get_backend("linear_model", "regression")

        if not hasattr(self, "_policy"):
            self._policy = self._get_policy(queue, X)

        if not hasattr(self, "_dtype"):
            self._dtype = get_dtype(X)
            self._params = self._get_onedal_params(self._dtype)

        if self._dtype not in [np.float32, np.float64]:
            self._dtype = np.float64

        X = X.astype(self._dtype, copy=self.copy_X)
        y = y.astype(dtype=self._dtype)
        self._y_ndim_1 = y.ndim == 1

        X, y = _check_X_y(X, y, force_all_finite=False, accept_2d_y=True)

        self.n_features_in_ = _num_features(X, fallback_1d=True)

        X, y = _convert_to_supported(self._policy, X, y)
        X_table, y_table = to_table(X, y)
        hparams = get_hyperparameters("linear_regression", "train")
        if hparams is not None and not hparams.is_default:
            self._partial_result = module.partial_train(
                self._policy,
                self._params,
                hparams.backend,
                self._partial_result,
                X_table,
                y_table,
            )
        else:
            self._partial_result = module.partial_train(
                self._policy, self._params, self._partial_result, X_table, y_table
            )

    def finalize_fit(self, queue=None):
        """
        Finalizes linear regression computation and obtains coefficients
        from the current `_partial_result`.

        Parameters
        ----------
        queue : dpctl.SyclQueue
            Not used here, added for API conformance

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        module = self._get_backend("linear_model", "regression")
        hparams = get_hyperparameters("linear_regression", "train")
        if hparams is not None and not hparams.is_default:
            result = module.finalize_train(
                self._policy, self._params, hparams.backend, self._partial_result
            )
        else:
            result = module.finalize_train(
                self._policy, self._params, self._partial_result
            )

        self._onedal_model = result.model

        packed_coefficients = from_table(result.model.packed_coefficients)
        self.coef_, self.intercept_ = (
            packed_coefficients[:, 1:],
            packed_coefficients[:, 0],
        )

        if self.coef_.shape[0] == 1 and self._y_ndim_1:
            self.coef_ = self.coef_.ravel()
            self.intercept_ = self.intercept_[0]

        return self
