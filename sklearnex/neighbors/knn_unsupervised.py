#!/usr/bin/env python
# ===============================================================================
# Copyright 2021 Intel Corporation
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

try:
    from packaging.version import Version
except ImportError:
    from distutils.version import LooseVersion as Version

import warnings
from numbers import Integral

import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.neighbors._ball_tree import BallTree
from sklearn.neighbors._base import VALID_METRICS
from sklearn.neighbors._base import NeighborsBase as sklearn_NeighborsBase
from sklearn.neighbors._kd_tree import KDTree
from sklearn.neighbors._unsupervised import NearestNeighbors as sklearn_NearestNeighbors
from sklearn.utils.validation import _deprecate_positional_args, check_is_fitted

from daal4py.sklearn._utils import control_n_jobs, run_with_n_jobs, sklearn_check_version
from onedal.neighbors import NearestNeighbors as onedal_NearestNeighbors
from onedal.utils import _check_array, _num_features, _num_samples

from .._device_offload import dispatch, wrap_output_data
from .common import KNeighborsDispatchingBase

if sklearn_check_version("0.22") and Version(sklearn_version) < Version("0.23"):

    class NearestNeighbors_(sklearn_NearestNeighbors):
        def __init__(
            self,
            n_neighbors=5,
            radius=1.0,
            algorithm="auto",
            leaf_size=30,
            metric="minkowski",
            p=2,
            metric_params=None,
            n_jobs=None,
        ):
            super().__init__(
                n_neighbors=n_neighbors,
                radius=radius,
                algorithm=algorithm,
                leaf_size=leaf_size,
                metric=metric,
                p=p,
                metric_params=metric_params,
                n_jobs=n_jobs,
            )

else:

    class NearestNeighbors_(sklearn_NearestNeighbors):
        if sklearn_check_version("1.2"):
            _parameter_constraints: dict = {
                **sklearn_NearestNeighbors._parameter_constraints
            }

        @_deprecate_positional_args
        def __init__(
            self,
            *,
            n_neighbors=5,
            radius=1.0,
            algorithm="auto",
            leaf_size=30,
            metric="minkowski",
            p=2,
            metric_params=None,
            n_jobs=None,
        ):
            super().__init__(
                n_neighbors=n_neighbors,
                radius=radius,
                algorithm=algorithm,
                leaf_size=leaf_size,
                metric=metric,
                p=p,
                metric_params=metric_params,
                n_jobs=n_jobs,
            )


@control_n_jobs
class NearestNeighbors(NearestNeighbors_, KNeighborsDispatchingBase):
    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {**NearestNeighbors_._parameter_constraints}

    @_deprecate_positional_args
    def __init__(
        self,
        n_neighbors=5,
        radius=1.0,
        algorithm="auto",
        leaf_size=30,
        metric="minkowski",
        p=2,
        metric_params=None,
        n_jobs=None,
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            radius=radius,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )

    def fit(self, X, y=None):
        self._fit_validation(X, y)
        dispatch(
            self,
            "fit",
            {
                "onedal": self.__class__._onedal_fit,
                "sklearn": sklearn_NearestNeighbors.fit,
            },
            X,
            None,
        )
        # Preserve original data for queue even if not correctly
        # formatted by 'check_array' or 'validate_data'.
        self._fit_X = X
        return self

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
        else:
            query_is_train = True
            X = self._fit_X
            n_neighbors += 1

        # _fit_X is not guaranteed to have been checked properly
        if sklearn_check_version("1.0"):
            self._check_feature_names(X, reset=False)

        result = dispatch(
            self,
            "kneighbors",
            {
                "onedal": self.__class__._onedal_kneighbors,
                "sklearn": sklearn_NearestNeighbors.kneighbors,
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
        distances = distances[:, 1:]
        indices = indices[:, 1:]

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

    @wrap_output_data
    def radius_neighbors(
        self, X=None, radius=None, return_distance=True, sort_results=False
    ):
        _onedal_estimator = getattr(self, "_onedal_estimator", None)

        if (
            _onedal_estimator is not None
            or getattr(self, "_tree", 0) is None
            and self._fit_method == "kd_tree"
        ):
            if sklearn_check_version("0.24"):
                sklearn_NearestNeighbors.fit(self, self._fit_X, getattr(self, "_y", None))
            else:
                sklearn_NearestNeighbors.fit(self, self._fit_X)
        if sklearn_check_version("0.22"):
            result = sklearn_NearestNeighbors.radius_neighbors(
                self, X, radius, return_distance, sort_results
            )
        else:
            result = sklearn_NearestNeighbors.radius_neighbors(
                self, X, radius, return_distance
            )

        return result

    @run_with_n_jobs
    def _onedal_fit(self, X, y=None, queue=None):
        onedal_params = {
            "n_neighbors": self.n_neighbors,
            "algorithm": self.algorithm,
            "metric": self.effective_metric_,
            "p": self.effective_metric_params_["p"],
        }

        try:
            requires_y = self._get_tags()["requires_y"]
        except KeyError:
            requires_y = False

        self._onedal_estimator = onedal_NearestNeighbors(**onedal_params)
        self._onedal_estimator.requires_y = requires_y
        self._onedal_estimator.effective_metric_ = self.effective_metric_
        self._onedal_estimator.effective_metric_params_ = self.effective_metric_params_
        self._onedal_estimator.fit(X, y, queue=queue)

        self._save_attributes()

    @run_with_n_jobs
    def _onedal_predict(self, X, queue=None):
        return self._onedal_estimator.predict(X, queue=queue)

    @run_with_n_jobs
    def _onedal_kneighbors(
        self, X=None, n_neighbors=None, return_distance=True, queue=None
    ):
        return self._onedal_estimator.kneighbors(
            X, n_neighbors, return_distance, queue=queue
        )

    def _save_attributes(self):
        self.classes_ = self._onedal_estimator.classes_
        self.n_features_in_ = self._onedal_estimator.n_features_in_
        self.n_samples_fit_ = self._onedal_estimator.n_samples_fit_
        self._fit_method = self._onedal_estimator._fit_method
        self._tree = self._onedal_estimator._tree
