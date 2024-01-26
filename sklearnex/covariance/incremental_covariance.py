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

import numbers

import numpy as np
from scipy import linalg
from sklearn.covariance import EmpiricalCovariance as sklearn_EmpericalCovariance
from sklearn.utils import check_array, gen_batches

from daal4py.sklearn._n_jobs_support import control_n_jobs
from daal4py.sklearn._utils import sklearn_check_version
from onedal._device_offload import support_usm_ndarray
from onedal.covariance import (
    IncrementalEmpiricalCovariance as onedal_IncrementalEmpiricalCovariance,
)
from sklearnex._device_offload import dispatch, wrap_output_data
from sklearnex._utils import PatchingConditionsChain, register_hyperparameters
from sklearnex.metrics import pairwise_distances

if sklearn_check_version("1.2"):
    from sklearn.utils._param_validation import Interval


# TODO: consult with others whether this should support store_precision and assume_centered
@control_n_jobs(decorated_methods=["partial_fit", "fit"])
class IncrementalEmpiricalCovariance(BaseEstimator):
    """
    Incremental estimator for covariance.
    Allows to compute empirical covariance estimated by maximum
    likelihood method if data are splitted into batches.

    Parameters
    ----------
    batch_size : int, default=None
        The number of samples to use for each batch. Only used when calling
        ``fit``. If ``batch_size`` is ``None``, then ``batch_size``
        is inferred from the data and set to ``5 * n_features``, to provide a
        balance between approximation accuracy and memory consumption.

    Attributes
    ----------
    location_ : ndarray of shape (n_features,)
        Estimated location, i.e. the estimated mean.

    covariance_ : ndarray of shape (n_features, n_features)
        Estimated covariance matrix
    """

    _onedal_incremental_covariance = staticmethod(onedal_IncrementalEmpiricalCovariance)

    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {
            "batch_size": [Interval(numbers.Integral, 1, None, closed="left")],
            "copy": ["boolean"],
        }

    def __init__(self, batch_size=None, copy=False):
        self._need_to_finalize = False  # If True then finalize compute should
        #      be called to obtain covariance_ or location_ from partial compute data
        self.batch_size = batch_size
        self.copy = copy

    def _onedal_supported(self, method_name, *data):
        patching_status = PatchingConditionsChain(
            "sklearn.covariance.{self.__class__.__name__}.{method_name}"
        )
        return patching_status

    def _onedal_finalize_fit(self):
        assert hasattr(self, "_onedal_estimator")
        self._onedal_estimator.finalize_fit()
        self.location_ = self._onedal_estimator.location_
        self.covariance_ = self._onedal_estimator.covariance_
        self._need_to_finalize = False

    def _onedal_partial_fit(self, X, queue):
        onedal_params = {
            "method": "dense",
            "bias": True,
        }
        if not hasattr(self, "_onedal_estimator"):
            self._onedal_estimator = self._onedal_incremental_covariance(**onedal_params)
        self._onedal_estimator.partial_fit(X, queue)
        self._need_to_finalize = True

    @property
    def covariance_(self):
        if hasattr(self, "_onedal_estimator"):
            if self._need_to_finalize:
                self._onedal_finalize_fit()
            return self._onedal_estimator.covariance_
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute 'covariance_'"
            )

    @property
    def location_(self):
        if hasattr(self, "_onedal_estimator"):
            if self._need_to_finalize:
                self._onedal_finalize_fit()
            return self._onedal_estimator.location_
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute 'location_'"
            )

    @support_usm_ndarray
    def partial_fit(self, X):
        """
        Incremental fit with X. All of X is processed as a single batch.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if sklearn_check_version("1.2"):
            X = self._validate_data(X, dtype=[np.float64, np.float32], copy=self.copy)
        else:
            X = check_array(X, dtype=[np.float64, np.float32], copy=self.copy)

        if not hasattr(self, "n_samples_seen_"):
            self.n_samples_seen_ = 0
        else:
            self.n_samples_seen_ += X.shape[0]

        dispatch(
            self,
            "partial_fit",
            {
                "onedal": self.__class__._onedal_partial_fit,
                "sklearn": None,
            },
            X,
        )

        return self

    @support_usm_ndarray
    def fit(self, X, y=None):
        dispatch(
            self,
            "fit",
            {
                "onedal": self.__class__._onedal_fit,
                "sklearn": None,
            },
            X,
        )

        return self

    def _onedal_fit(self, X, queue=None):
        """
        Fit the model with X, using minibatches of size batch_size.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.n_samples_seen_ = 0

        if sklearn_check_version("1.2"):
            self._validate_params()
            X = self._validate_data(X, dtype=[np.float64, np.float32], copy=self.copy)
        else:
            X = check_array(X, dtype=[np.float64, np.float32], copy=self.copy)

        n_samples, self.n_features_in_ = X.shape

        if self.batch_size is None:
            self.batch_size_ = 5 * self.n_features_in_
        else:
            self.batch_size_ = self.batch_size
        for batch in gen_batches(n_samples, batch_size_):
            X_batch = X[batch]
            self._onedal_partial_fit(X_batch, queue=queue)
            self.n_samples_seen_ += X.shape[0]

        self._onedal_finalize_fit()
        return self

    get_precision = sklearn_EmpiricalCovariance.get_precision
    error_norm = wrap_output_data(sklearn_EmpericalCovariance.error_norm)

    # necessary to to use sklearnex pairwise_distances
    def _onedal_mahalanobis(self, X):
        if sklearn_check_version("1.2"):
            X = self._validate_data(X, reset=False, copy=self.copy)
        else:
            X = check_array(X, reset=False, copy=self.copy)

        precision = self.get_precision()
        with config_context(assume_finite=True):
            # compute mahalanobis distances
            dist = pairwise_distances(
                X, self.location_[np.newaxis, :], metric="mahalanobis", VI=precision
            )

        return np.reshape(dist, (len(X),)) ** 2

    @wrap_output_data
    def mahalanobis(self, X):
        dispatch(
            self,
            "mahalanobis",
            {
                "onedal": self.__class__._onedal_mahalanobis,
                "sklearn": sklearn_EmpericalCovariance.mahalanobis,
            },
            X,
        )
        return self

    _onedal_cpu_supported = _onedal_supported
    _onedal_gpu_supported = _onedal_supported

    fit.__doc__ = sklearn_EmpiricalCovariance.fit.__doc__
    mahalanobis.__doc__ = sklearn_EmpiricalCovariance.mahalanobis.__doc__
    error_norm.__doc__ = sklearn_EmpiricalCovariance.error_norm.__doc__
