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

from daal4py.sklearn._utils import control_n_jobs, run_with_n_jobs
from onedal._device_offload import support_usm_ndarray
from onedal.basic_statistics import BasicStatistics as onedal_BasicStatistics


@control_n_jobs
class BasicStatistics:
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

        for option in result_options:
            setattr(self, option, getattr(self._onedal_estimator, option))

    @run_with_n_jobs
    def _onedal_fit(self, X, weights=None, queue=None):
        onedal_params = {
            "algorithm": "by_default",
            "result_options": self.options,
        }
        if not hasattr(self, "_onedal_estimator"):
            self._onedal_estimator = self._onedal_basic_statistics(**onedal_params)
        self._onedal_estimator.fit(X, weights, queue)
        self._save_attributes()

    @support_usm_ndarray()
    def fit(self, X, weights=None, queue=None):
        """Fit with X.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data for compute, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        weights : array-like of shape (n_samples,)
            Weights for compute weighted statistics, where `n_samples` is the number of samples.
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._onedal_fit(X, weights, queue)
        return self
