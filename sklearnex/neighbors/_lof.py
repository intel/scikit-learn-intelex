#!/usr/bin/env python
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

import warnings

import numpy as np
from sklearn.neighbors import LocalOutlierFactor as sklearn_LocalOutlierFactor
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

from daal4py.sklearn._utils import control_n_jobs, run_with_n_jobs, sklearn_check_version

from .._device_offload import dispatch, wrap_output_data
from .common import KNeighborsDispatchingBase
from .knn_unsupervised import NearestNeighbors


@control_n_jobs
class LocalOutlierFactor(KNeighborsDispatchingBase, sklearn_LocalOutlierFactor):
    __doc__ = (
        sklearn_LocalOutlierFactor.__doc__
        + "NOTE: When X=None, methods kneighbors, kneighbors_graph, and predict will"
        + "\n only output numpy arrays. In that case, the only way to offload to gpu"
        + "\n is to use a global queue (e.g. using config_context)"
    )
    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {
            **sklearn_LocalOutlierFactor._parameter_constraints
        }

    # Only certain methods should be taken from knn to prevent code
    # duplication. Inheriting would yield a complicated inheritance
    # structure, wrap
    _save_attributes = NearestNeighbors._save_attributes
    _onedal_knn_fit = NearestNeighbors._onedal_fit

    @run_with_n_jobs
    def _onedal_fit(self, X, y, queue=None):
        if sklearn_check_version("1.2"):
            self._validate_params()

        self._onedal_knn_fit(X, y, queue)

        if self.contamination != "auto":
            if not (0.0 < self.contamination <= 0.5):
                raise ValueError(
                    "contamination must be in (0, 0.5], " "got: %f" % self.contamination
                )

        n_samples = self.n_samples_fit_

        if self.n_neighbors > n_samples:
            warnings.warn(
                "n_neighbors (%s) is greater than the "
                "total number of samples (%s). n_neighbors "
                "will be set to (n_samples - 1) for estimation."
                % (self.n_neighbors, n_samples)
            )
        self.n_neighbors_ = max(1, min(self.n_neighbors, n_samples - 1))

        (
            self._distances_fit_X_,
            _neighbors_indices_fit_X_,
        ) = self._onedal_estimator.kneighbors(n_neighbors=self.n_neighbors_, queue=queue)

        # Sklearn includes a check for float32 at this point which may not be
        # necessary for onedal

        self._lrd = self._local_reachability_density(
            self._distances_fit_X_, _neighbors_indices_fit_X_
        )

        # Compute lof score over training samples to define offset_:
        lrd_ratios_array = self._lrd[_neighbors_indices_fit_X_] / self._lrd[:, np.newaxis]

        self.negative_outlier_factor_ = -np.mean(lrd_ratios_array, axis=1)

        if self.contamination == "auto":
            # inliers score around -1 (the higher, the less abnormal).
            self.offset_ = -1.5
        else:
            self.offset_ = np.percentile(
                self.negative_outlier_factor_, 100.0 * self.contamination
            )

        return self

    def fit(self, X, y=None):
        self._fit_validation(X, y)
        result = dispatch(
            self,
            "fit",
            {
                "onedal": self.__class__._onedal_fit,
                "sklearn": sklearn_LocalOutlierFactor.fit,
            },
            X,
            None,
        )
        return result

    # Subtle order change to remove check_array and preserve dpnp and
    # dpctl conformance. decision_function will return a dpnp or dpctl
    # instance via kneighbors and an equivalent check_array exists in
    # that call already in sklearn so no loss of functionality occurs
    def _predict(self, X=None):
        check_is_fitted(self)

        if X is not None:
            output = self.decision_function(X) < 0
            is_inlier = np.ones(output.shape[0], dtype=int)
            is_inlier[output] = -1
        else:
            is_inlier = np.ones(self.n_samples_fit_, dtype=int)
            is_inlier[self.negative_outlier_factor_ < self.offset_] = -1

        return is_inlier

    # This had to be done because predict loses the queue when no
    # argument is given and it is a dpctl tensor or dpnp array.
    # This would cause issues in fit_predict. Also, available_if
    # is hard to unwrap, and this is the most straighforward way.
    @available_if(sklearn_LocalOutlierFactor._check_novelty_fit_predict)
    @wrap_output_data
    def fit_predict(self, X, y=None):
        return self.fit(X)._predict()

    @available_if(sklearn_LocalOutlierFactor._check_novelty_predict)
    @wrap_output_data
    def predict(self, X=None):
        return self._predict(X)

    @run_with_n_jobs
    def _onedal_score_samples(self, X, queue=None):
        distances_X, neighbors_indices_X = self._onedal_estimator._kneighbors(
            X, n_neighbors=self.n_neighbors_, queue=queue
        )

        X_lrd = self._local_reachability_density(
            distances_X,
            neighbors_indices_X,
        )

        lrd_ratios_array = self._lrd[neighbors_indices_X] / X_lrd[:, np.newaxis]

        # as bigger is better:
        return -np.mean(lrd_ratios_array, axis=1)

    # Only necessary to preserve dpnp and dpctl conformance, otherwise a copy
    @available_if(sklearn_LocalOutlierFactor._check_novelty_score_samples)
    @wrap_output_data
    def score_samples(self, X):
        check_is_fitted(self)
        if sklearn_check_version("1.0") and X is not None:
            self._check_feature_names(X, reset=False)
        return dispatch(
            self,
            "score_samples",
            {
                "onedal": self.__class__._onedal_score_samples,
                "sklearn": sklearn_LocalOutlierFactor.score_samples,
            },
            X,
        )

    @wrap_output_data
    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        check_is_fitted(self)
        if sklearn_check_version("1.0") and X is not None:
            self._check_feature_names(X, reset=False)
        return dispatch(
            self,
            "kneighbors",
            {
                "onedal": NearestNeighbors._onedal_kneighbors,
                "sklearn": sklearn_LocalOutlierFactor.kneighbors,
            },
            X,
            n_neighbors,
            return_distance,
        )

    fit.__doc__ = sklearn_LocalOutlierFactor.fit.__doc__
    fit_predict.__doc__ = sklearn_LocalOutlierFactor.fit_predict.__doc__
    predict.__doc__ = sklearn_LocalOutlierFactor.predict.__doc__
    score_samples.__doc__ = sklearn_LocalOutlierFactor.score_samples.__doc__
    kneighbors.__doc__ = sklearn_LocalOutlierFactor.kneighbors.__doc__
