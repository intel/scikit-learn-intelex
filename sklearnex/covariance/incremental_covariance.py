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
import warnings

import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator
from sklearn.covariance import EmpiricalCovariance as sklearn_EmpiricalCovariance
from sklearn.utils import check_array, gen_batches

from daal4py.sklearn._n_jobs_support import control_n_jobs
from daal4py.sklearn._utils import daal_check_version, sklearn_check_version
from onedal._device_offload import support_usm_ndarray
from onedal.covariance import (
    IncrementalEmpiricalCovariance as onedal_IncrementalEmpiricalCovariance,
)
from sklearnex import config_context

from .._device_offload import dispatch, wrap_output_data
from .._utils import PatchingConditionsChain, register_hyperparameters
from ..metrics import pairwise_distances

if sklearn_check_version("1.2"):
    from sklearn.utils._param_validation import Interval


@control_n_jobs(decorated_methods=["partial_fit", "fit", "_onedal_finalize_fit"])
class IncrementalEmpiricalCovariance(BaseEstimator):
    """
    Incremental estimator for covariance.
    Allows to compute empirical covariance estimated by maximum
    likelihood method if data are splitted into batches.

    Parameters
    ----------
    store_precision : bool, default=False
        Specifies if the estimated precision is stored.

    assume_centered : bool, default=False
        If True, data are not centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False (default), data are centered before computation.

    batch_size : int, default=None
        The number of samples to use for each batch. Only used when calling
        ``fit``. If ``batch_size`` is ``None``, then ``batch_size``
        is inferred from the data and set to ``5 * n_features``, to provide a
        balance between approximation accuracy and memory consumption.

    copy : bool, default=True
        If False, X will be overwritten. ``copy=False`` can be used to
        save memory but is unsafe for general use.

    Attributes
    ----------
    location_ : ndarray of shape (n_features,)
        Estimated location, i.e. the estimated mean.

    covariance_ : ndarray of shape (n_features, n_features)
        Estimated covariance matrix

    n_samples_seen_ : int
        The number of samples processed by the estimator. Will be reset on
        new calls to fit, but increments across ``partial_fit`` calls.

    batch_size_ : int
        Inferred batch size from ``batch_size``.

    n_features_in_ : int
        Number of features seen during :term:`fit` `partial_fit`.
    """

    _onedal_incremental_covariance = staticmethod(onedal_IncrementalEmpiricalCovariance)

    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {
            "store_precision": ["boolean"],
            "assume_centered": ["boolean"],
            "batch_size": [Interval(numbers.Integral, 1, None, closed="left"), None],
            "copy": ["boolean"],
        }

    get_precision = sklearn_EmpiricalCovariance.get_precision
    error_norm = wrap_output_data(sklearn_EmpiricalCovariance.error_norm)
    score = wrap_output_data(sklearn_EmpiricalCovariance.score)

    def __init__(
        self, *, store_precision=False, assume_centered=False, batch_size=None, copy=True
    ):
        self.assume_centered = assume_centered
        self.store_precision = store_precision
        self.batch_size = batch_size
        self.copy = copy

    def _onedal_supported(self, method_name, *data):
        patching_status = PatchingConditionsChain(
            f"sklearn.covariance.{self.__class__.__name__}.{method_name}"
        )
        return patching_status

    def _onedal_finalize_fit(self):
        assert hasattr(self, "_onedal_estimator")
        self._onedal_estimator.finalize_fit()
        self._need_to_finalize = False

        if not daal_check_version((2024, "P", 400)) and self.assume_centered:
            location = self._onedal_estimator.location_[None, :]
            self._onedal_estimator.covariance_ += np.dot(location.T, location)
            self._onedal_estimator.location_ = np.zeros_like(np.squeeze(location))
        if self.store_precision:
            self.precision_ = linalg.pinvh(
                self._onedal_estimator.covariance_, check_finite=False
            )
        else:
            self.precision_ = None

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

    def _onedal_partial_fit(self, X, queue=None, check_input=True):

        first_pass = not hasattr(self, "n_samples_seen_") or self.n_samples_seen_ == 0

        # finite check occurs on onedal side
        if check_input:
            if sklearn_check_version("1.2"):
                self._validate_params()

            if sklearn_check_version("1.0"):
                X = self._validate_data(
                    X,
                    dtype=[np.float64, np.float32],
                    reset=first_pass,
                    copy=self.copy,
                    force_all_finite=False,
                )
            else:
                X = check_array(
                    X,
                    dtype=[np.float64, np.float32],
                    copy=self.copy,
                    force_all_finite=False,
                )

        onedal_params = {
            "method": "dense",
            "bias": True,
            "assume_centered": self.assume_centered,
        }
        if not hasattr(self, "_onedal_estimator"):
            self._onedal_estimator = self._onedal_incremental_covariance(**onedal_params)
        try:
            if first_pass:
                self.n_samples_seen_ = X.shape[0]
                self.n_features_in_ = X.shape[1]
            else:
                self.n_samples_seen_ += X.shape[0]

            self._onedal_estimator.partial_fit(X, queue)
        finally:
            self._need_to_finalize = True

        return self

    def partial_fit(self, X, y=None, check_input=True):
        """
        Incremental fit with X. All of X is processed as a single batch.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        check_input : bool, default=True
            Run check_array on X.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        return dispatch(
            self,
            "partial_fit",
            {
                "onedal": self.__class__._onedal_partial_fit,
                "sklearn": None,
            },
            X,
            check_input=check_input,
        )

    def fit(self, X, y=None):
        """
        Fit the model with X, using minibatches of size batch_size.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        return dispatch(
            self,
            "fit",
            {
                "onedal": self.__class__._onedal_fit,
                "sklearn": None,
            },
            X,
        )

    def _onedal_fit(self, X, queue=None):
        self.n_samples_seen_ = 0
        if hasattr(self, "_onedal_estimator"):
            self._onedal_estimator._reset()

        if sklearn_check_version("1.2"):
            self._validate_params()

        # finite check occurs on onedal side
        if sklearn_check_version("1.0"):
            X = self._validate_data(
                X, dtype=[np.float64, np.float32], copy=self.copy, force_all_finite=False
            )
        else:
            X = check_array(
                X, dtype=[np.float64, np.float32], copy=self.copy, force_all_finite=False
            )
            self.n_features_in_ = X.shape[1]

        self.batch_size_ = self.batch_size if self.batch_size else 5 * self.n_features_in_

        if X.shape[0] == 1:
            warnings.warn(
                "Only one sample available. You may want to reshape your data array"
            )

        for batch in gen_batches(X.shape[0], self.batch_size_):
            X_batch = X[batch]
            self._onedal_partial_fit(X_batch, queue=queue, check_input=False)

        self._onedal_finalize_fit()

        return self

    # expose sklearnex pairwise_distances if mahalanobis distance eventually supported
    @wrap_output_data
    def mahalanobis(self, X):
        if sklearn_check_version("1.0"):
            self._validate_data(X, reset=False, copy=self.copy)
        else:
            check_array(X, copy=self.copy)

        precision = self.get_precision()
        with config_context(assume_finite=True):
            # compute mahalanobis distances
            dist = pairwise_distances(
                X, self.location_[np.newaxis, :], metric="mahalanobis", VI=precision
            )

        return np.reshape(dist, (len(X),)) ** 2

    _onedal_cpu_supported = _onedal_supported
    _onedal_gpu_supported = _onedal_supported

    mahalanobis.__doc__ = sklearn_EmpiricalCovariance.mahalanobis.__doc__
    error_norm.__doc__ = sklearn_EmpiricalCovariance.error_norm.__doc__
    score.__doc__ = sklearn_EmpiricalCovariance.score.__doc__
