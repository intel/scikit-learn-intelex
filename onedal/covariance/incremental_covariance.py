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
import numpy as np

from daal4py.sklearn._utils import daal_check_version, get_dtype

from ..datatypes import _convert_to_supported, from_table, to_table
from ..utils import _check_array
from .covariance import BaseEmpiricalCovariance


class IncrementalEmpiricalCovariance(BaseEmpiricalCovariance):
    """
    Covariance estimator based on oneDAL implementation.

    Computes sample covariance matrix.

    Parameters
    ----------
    method : string, default="dense"
        Specifies computation method. Available methods: "dense".

    bias: bool, default=False
        If True biased estimation of covariance is computed which equals to
        the unbiased one multiplied by (n_samples - 1) / n_samples.

    assume_centered : bool, default=False
        If True, data are not centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False (default), data are centered before computation.

    Attributes
    ----------
    location_ : ndarray of shape (n_features,)
        Estimated location, i.e., the estimated mean.

    covariance_ : ndarray of shape (n_features, n_features)
        Estimated covariance matrix
    """

    def __init__(self, method="dense", bias=False, assume_centered=False):
        super().__init__(method, bias, assume_centered)
        self._reset()

    def _reset(self):
        self._need_to_finalize = False
        self._partial_result = self._get_backend(
            "covariance", None, "partial_compute_result"
        )

    def __getstate__(self):
        self.finalize_fit()
        data = self.__dict__.copy()
        partial_result_data = dict()
        partial_result_data["partial_n_rows"] = from_table(
            data["_partial_result"].partial_n_rows
        )
        partial_result_data["partial_crossproduct"] = from_table(
            data["_partial_result"].partial_crossproduct
        )
        partial_result_data["partial_sums"] = from_table(
            data["_partial_result"].partial_sums
        )
        data["_partial_result"] = partial_result_data
        data.pop("_policy", None)

        return data

    def __setstate__(self, data):
        partial_result = self._get_backend("covariance", None, "partial_compute_result")
        if data["_partial_result"]["partial_n_rows"].size > 0:
            partial_result.partial_n_rows = to_table(
                data["_partial_result"]["partial_n_rows"]
            )
        if data["_partial_result"]["partial_crossproduct"].size > 0:
            partial_result.partial_crossproduct = to_table(
                data["_partial_result"]["partial_crossproduct"]
            )
        if data["_partial_result"]["partial_sums"].size > 0:
            partial_result.partial_sums = to_table(
                data["_partial_result"]["partial_sums"]
            )

        data["_partial_result"] = partial_result

        self.__dict__ = data

    def partial_fit(self, X, y=None, queue=None):
        """
        Computes partial data for the covariance matrix
        from data batch X and saves it to `_partial_result`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data batch, where `n_samples` is the number of samples
            in the batch, and `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        queue : dpctl.SyclQueue
            If not None, use this queue for computations.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = _check_array(X, dtype=[np.float64, np.float32], ensure_2d=True)

        self._queue = queue

        policy = self._get_policy(queue, X)

        X = _convert_to_supported(policy, X)

        if not hasattr(self, "_dtype"):
            self._dtype = get_dtype(X)

        params = self._get_onedal_params(self._dtype)
        table_X = to_table(X)
        self._partial_result = self._get_backend(
            "covariance",
            None,
            "partial_compute",
            policy,
            params,
            self._partial_result,
            table_X,
        )
        self._need_to_finalize = True

    def finalize_fit(self, queue=None):
        """
        Finalizes covariance matrix and obtains `covariance_` and `location_`
        attributes from the current `_partial_result`.

        Parameters
        ----------
        queue : dpctl.SyclQueue
            If not None, use this queue for computations.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        params = self._get_onedal_params(self._dtype)
        if queue is not None:
            policy = self._get_policy(queue)
        else:
            policy = self._get_policy(self._queue)

        result = self._get_backend(
            "covariance",
            None,
            "finalize_compute",
            policy,
            params,
            self._partial_result,
        )
        if daal_check_version((2024, "P", 1)) or (not self.bias):
            self.covariance_ = from_table(result.cov_matrix)
        else:
            n_rows = self._partial_result.partial_n_rows
            self.covariance_ = from_table(result.cov_matrix) * (n_rows - 1) / n_rows

            self.location_ = from_table(result.means).ravel()

        self._need_to_finalize = False

        return self
