# ==============================================================================
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
# ==============================================================================

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import _check_sample_weight

from daal4py.sklearn._n_jobs_support import control_n_jobs
from daal4py.sklearn._utils import sklearn_check_version
from onedal.basic_statistics import BasicStatistics as onedal_BasicStatistics

from .._device_offload import dispatch
from .._utils import PatchingConditionsChain


@control_n_jobs(decorated_methods=["fit"])
class BasicStatistics(BaseEstimator):
    """
    Estimator for basic statistics.
    Allows to compute basic statistics for provided data.
    Parameters
    ----------
    result_options: string or list, default='all'
        List of statistics to compute

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

    def __init__(self, result_options="all"):
        self.options = result_options

    _onedal_basic_statistics = staticmethod(onedal_BasicStatistics)

    def _save_attributes(self):
        assert hasattr(self, "_onedal_estimator")

        if self.options == "all":
            result_options = onedal_BasicStatistics.get_all_result_options()
        else:
            result_options = self.options

        if isinstance(result_options, str):
            setattr(self, result_options, getattr(self._onedal_estimator, result_options))
        elif isinstance(result_options, list):
            for option in result_options:
                setattr(self, option, getattr(self._onedal_estimator, option))

    def _onedal_supported(self, method_name, *data):
        patching_status = PatchingConditionsChain(
            f"sklearnex.basic_statistics.{self.__class__.__name__}.{method_name}"
        )
        return patching_status

    _onedal_cpu_supported = _onedal_supported
    _onedal_gpu_supported = _onedal_supported

    def _onedal_fit(self, X, sample_weight=None, queue=None):
        if sklearn_check_version("1.0"):
            X = self._validate_data(X, dtype=[np.float64, np.float32], ensure_2d=False)
        else:
            X = check_array(X, dtype=[np.float64, np.float32])

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        onedal_params = {
            "result_options": self.options,
        }

        if not hasattr(self, "_onedal_estimator"):
            self._onedal_estimator = self._onedal_basic_statistics(**onedal_params)
        self._onedal_estimator.fit(X, sample_weight, queue)
        self._save_attributes()

    def compute(self, data, weights=None, queue=None):
        return self._onedal_estimator.compute(data, weights, queue)

    def fit(self, X, y=None, *, sample_weight=None):
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
