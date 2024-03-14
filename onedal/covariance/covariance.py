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
from abc import ABCMeta

import numpy as np

from daal4py.sklearn._utils import daal_check_version, get_dtype
from onedal.utils import _check_array

from ..common._base import BaseEstimator
from ..common.hyperparameters import get_hyperparameters
from ..datatypes import _convert_to_supported, from_table, to_table


class BaseEmpiricalCovariance(BaseEstimator, metaclass=ABCMeta):
    def __init__(self, method="dense", bias=False):
        self.method = method
        self.bias = bias

    def _get_onedal_params(self, dtype=np.float32):
        params = {
            "fptype": "float" if dtype == np.float32 else "double",
            "method": self.method,
        }
        if daal_check_version((2024, "P", 1)):
            params["bias"] = self.bias

        return params


class EmpiricalCovariance(BaseEmpiricalCovariance):
    """Covariance estimator.

    Computes sample covariance matrix.

    Parameters
    ----------
    method : string, default="dense"
        Specifies computation method. Available methods: "dense".

    bias: bool, default=False
        If True biased estimation of covariance is computed which equals to
        the unbiased one multiplied by (n_samples - 1) / n_samples.

    Attributes
    ----------
    location_ : ndarray of shape (n_features,)
        Estimated location, i.e., the estimated mean.

    covariance_ : ndarray of shape (n_features, n_features)
        Estimated covariance matrix
    """

    def fit(self, X, y=None, queue=None):
        """Fit the sample covariance matrix of X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples, and
            `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        queue : dpctl.SyclQueue
            If not None, use this queue for computations.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        policy = self._get_policy(queue, X)
        X = _check_array(X, dtype=[np.float64, np.float32])
        X = _convert_to_supported(policy, X)
        dtype = get_dtype(X)
        params = self._get_onedal_params(dtype)
        hparams = get_hyperparameters("covariance", "compute")
        if hparams is not None and not hparams.is_default:
            result = self._get_backend(
                "covariance",
                None,
                "compute",
                policy,
                params,
                hparams.backend,
                to_table(X),
            )
        else:
            result = self._get_backend(
                "covariance", None, "compute", policy, params, to_table(X)
            )
        if daal_check_version((2024, "P", 1)) or (not self.bias):
            self.covariance_ = from_table(result.cov_matrix)
        else:
            self.covariance_ = (
                from_table(result.cov_matrix) * (X.shape[0] - 1) / X.shape[0]
            )

        self.location_ = from_table(result.means).ravel()

        return self
