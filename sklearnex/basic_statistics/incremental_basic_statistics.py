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
from sklearn.base import BaseEstimator
from sklearn.utils import check_array, gen_batches
from sklearn.utils.validation import _check_sample_weight

from daal4py.sklearn._n_jobs_support import control_n_jobs
from daal4py.sklearn._utils import sklearn_check_version
from onedal.basic_statistics import (
    IncrementalBasicStatistics as onedal_IncrementalBasicStatistics,
)

from .._device_offload import dispatch
from .._utils import PatchingConditionsChain

if sklearn_check_version("1.2"):
    from sklearn.utils._param_validation import Interval, StrOptions

import numbers


@control_n_jobs(decorated_methods=["partial_fit", "_onedal_finalize_fit"])
class IncrementalBasicStatistics(BaseEstimator):
    """
    Incremental estimator for basic statistics.
    Allows to compute basic statistics if data are splitted into batches.
    Parameters
    ----------
    result_options: string or list, default='all'
        List of statistics to compute

    batch_size : int, default=None
        The number of samples to use for each batch. Only used when calling
        ``fit``. If ``batch_size`` is ``None``, then ``batch_size``
        is inferred from the data and set to ``5 * n_features``, to provide a
        balance between approximation accuracy and memory consumption.

    Attributes (are existing only if corresponding result option exists)
    ----------
        min : ndarray of shape (n_features,)
            Minimum of each feature over all samples.

        max : ndarray of shape (n_features,)
            Maximum of each feature over all samples.

        sum : ndarray of shape (n_features,)
            Sum of each feature over all samples.

        mean : ndarray of shape (n_features,)
            Mean of each feature over all samples.

        variance : ndarray of shape (n_features,)
            Variance of each feature over all samples.

        variation : ndarray of shape (n_features,)
            Variation of each feature over all samples.

        sum_squares : ndarray of shape (n_features,)
            Sum of squares for each feature over all samples.

        standard_deviation : ndarray of shape (n_features,)
            Standard deviation of each feature over all samples.

        sum_squares_centered : ndarray of shape (n_features,)
            Centered sum of squares for each feature over all samples.

        second_order_raw_moment : ndarray of shape (n_features,)
            Second order moment of each feature over all samples.
    """

    _onedal_incremental_basic_statistics = staticmethod(onedal_IncrementalBasicStatistics)

    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {
            "result_options": [
                StrOptions(
                    {
                        "all",
                        "min",
                        "max",
                        "sum",
                        "mean",
                        "variance",
                        "variation",
                        "sum_squares",
                        "standard_deviation",
                        "sum_squares_centered",
                        "second_order_raw_moment",
                    }
                ),
                list,
            ],
            "batch_size": [Interval(numbers.Integral, 1, None, closed="left"), None],
        }

    def __init__(self, result_options="all", batch_size=None):
        if result_options == "all":
            self.result_options = (
                self._onedal_incremental_basic_statistics.get_all_result_options()
            )
        else:
            self.result_options = result_options
        self._need_to_finalize = False
        self.batch_size = batch_size

    def _onedal_supported(self, method_name, *data):
        patching_status = PatchingConditionsChain(
            f"sklearn.covariance.{self.__class__.__name__}.{method_name}"
        )
        return patching_status

    _onedal_cpu_supported = _onedal_supported
    _onedal_gpu_supported = _onedal_supported

    def _get_onedal_result_options(self, options):
        if isinstance(options, list):
            onedal_options = "|".join(self.result_options)
        else:
            onedal_options = options
        assert isinstance(onedal_options, str)
        return options

    def _onedal_finalize_fit(self):
        assert hasattr(self, "_onedal_estimator")
        self._onedal_estimator.finalize_fit()
        self._need_to_finalize = False

    def _onedal_partial_fit(self, X, sample_weight=None, queue=None):
        first_pass = not hasattr(self, "n_samples_seen_") or self.n_samples_seen_ == 0

        if sklearn_check_version("1.0"):
            X = self._validate_data(
                X,
                dtype=[np.float64, np.float32],
                reset=first_pass,
            )
        else:
            X = check_array(
                X,
                dtype=[np.float64, np.float32],
            )

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        if first_pass:
            self.n_samples_seen_ = X.shape[0]
            self.n_features_in_ = X.shape[1]
        else:
            self.n_samples_seen_ += X.shape[0]

        onedal_params = {
            "result_options": self._get_onedal_result_options(self.result_options)
        }
        if not hasattr(self, "_onedal_estimator"):
            self._onedal_estimator = self._onedal_incremental_basic_statistics(
                **onedal_params
            )
        self._onedal_estimator.partial_fit(X, sample_weight, queue)
        self._need_to_finalize = True

    def _onedal_fit(self, X, sample_weight=None, queue=None):
        if sklearn_check_version("1.0"):
            X = self._validate_data(X, dtype=[np.float64, np.float32])
        else:
            X = check_array(X, dtype=[np.float64, np.float32])

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        n_samples, n_features = X.shape
        if self.batch_size is None:
            self.batch_size_ = 5 * n_features
        else:
            self.batch_size_ = self.batch_size

        self.n_samples_seen_ = 0
        if hasattr(self, "_onedal_estimator"):
            self._onedal_estimator._reset()

        for batch in gen_batches(X.shape[0], self.batch_size_):
            X_batch = X[batch]
            weights_batch = sample_weight[batch] if sample_weight is not None else None
            self._onedal_partial_fit(X_batch, weights_batch, queue=queue)

        if sklearn_check_version("1.2"):
            self._validate_params()

        self.n_features_in_ = X.shape[1]

        self._onedal_finalize_fit()

        return self

    def __getattr__(self, attr):
        result_options = self.__dict__["result_options"]
        is_statistic_attr = (
            isinstance(result_options, str) and (attr == result_options)
        ) or (isinstance(result_options, list) and (attr in result_options))
        if is_statistic_attr:
            if self._need_to_finalize:
                self._onedal_finalize_fit()
            return getattr(self._onedal_estimator, attr)
        if attr in self.__dict__:
            return self.__dict__[attr]

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{attr}'"
        )

    def partial_fit(self, X, sample_weight=None):
        """Incremental fit with X. All of X is processed as a single batch.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data for compute, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights for compute weighted statistics, where `n_samples` is the number of samples.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        dispatch(
            self,
            "partial_fit",
            {
                "onedal": self.__class__._onedal_partial_fit,
                "sklearn": None,
            },
            X,
            sample_weight,
        )
        return self

    def fit(self, X, y=None, sample_weight=None):
        """Compute statistics with X, using minibatches of size batch_size.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data for compute, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights for compute weighted statistics, where `n_samples` is the number of samples.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        dispatch(
            self,
            "fit",
            {
                "onedal": self.__class__._onedal_fit,
                "sklearn": None,
            },
            X,
            sample_weight,
        )
        return self
