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
from numbers import Integral

import numpy as np
from sklearn.neighbors import LocalOutlierFactor as sklearn_LocalOutlierFactor
from sklearn.utils.validation import check_is_fitted

from daal4py.sklearn._utils import control_n_jobs, run_with_n_jobs, sklearn_check_version

from .._device_offload import dispatch, wrap_output_data
from .common import KNeighborsDispatchingBase
from .knn_unsupervised import NearestNeighbors


@control_n_jobs
class LocalOutlierFactor(KNeighborsDispatchingBase, sklearn_LocalOutlierFactor):
    __doc__ = sklearn_LocalOutlierFactor.__doc__
    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {
            **sklearn_LocalOutlierFactor._parameter_constraints
        }

    # Only certain methods should be taken from knn to prevent code
    # duplication. Inheriting would yield a complicated inheritance
    # structure
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
            y,
        )
        # Preserve queue even if not correctly
        # formatted by 'check_array' or 'validate_data'.
        self._fit_X = X
        return result

    @wrap_output_data
    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        check_is_fitted(self)

        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        elif n_neighbors <= 0:
            raise ValueError("Expected n_neighbors > 0. Got %d" % n_neighbors)
        else:
            if not isinstance(n_neighbors, Integral):
                raise TypeError(
                    "n_neighbors does not take %s value, "
                    "enter integer value" % type(n_neighbors)
                )

        if X is not None:
            query_is_train = False
            if sklearn_check_version("1.0"):
                self._check_feature_names(X, reset=False)
        else:
            query_is_train = True
            X = self._fit_X
            n_neighbors += 1

        results = dispatch(
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

        if not query_is_train:
            return results
        # If the query data is the same as the indexed data, we would like
        # to ignore the first nearest neighbor of every sample, i.e
        # the sample itself.
        if return_distance:
            neigh_dist, neigh_ind = results
        else:
            neigh_ind = results

        n_queries, _ = X.shape
        sample_range = np.arange(n_queries)[:, None]
        sample_mask = neigh_ind != sample_range

        # Corner case: When the number of duplicates are more
        # than the number of neighbors, the first NN will not
        # be the sample, but a duplicate.
        # In that case mask the first duplicate.
        dup_gr_nbrs = np.all(sample_mask, axis=1)
        sample_mask[:, 0][dup_gr_nbrs] = False

        neigh_ind = np.reshape(neigh_ind[sample_mask], (n_queries, n_neighbors - 1))

        if return_distance:
            neigh_dist = np.reshape(neigh_dist[sample_mask], (n_queries, n_neighbors - 1))
            return neigh_dist, neigh_ind
        return neigh_ind

    fit.__doc__ = sklearn_LocalOutlierFactor.fit.__doc__
    kneighbors.__doc__ = sklearn_LocalOutlierFactor.kneighbors.__doc__
