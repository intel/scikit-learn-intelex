#!/usr/bin/env python
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

import warnings

import numpy as np
from sklearn.neighbors._lof import LocalOutlierFactor as sklearn_LocalOutlierFactor

from .knn_unsupervised import NearestNeighbors

try:
    from sklearn.utils.metaestimators import available_if
except ImportError:
    pass

from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from daal4py.sklearn._utils import sklearn_check_version

from .._config import config_context
from .._device_offload import dispatch, wrap_output_data
from .._utils import PatchingConditionsChain

if sklearn_check_version("1.0"):

    class LocalOutlierFactor(sklearn_LocalOutlierFactor):
        if sklearn_check_version("1.2"):
            _parameter_constraints: dict = {
                **sklearn_LocalOutlierFactor._parameter_constraints
            }

        def __init__(
            self,
            n_neighbors=20,
            *,
            algorithm="auto",
            leaf_size=30,
            metric="minkowski",
            p=2,
            metric_params=None,
            contamination="auto",
            novelty=False,
            n_jobs=None,
        ):
            super().__init__(
                n_neighbors=n_neighbors,
                algorithm=algorithm,
                leaf_size=leaf_size,
                metric=metric,
                p=p,
                metric_params=metric_params,
                n_jobs=n_jobs,
                contamination=contamination,
                novelty=novelty,
            )

        def _fit(self, X, y, queue=None):
            with config_context(target_offload=queue):
                if sklearn_check_version("1.2"):
                    self._validate_params()
                self._knn = NearestNeighbors(
                    n_neighbors=self.n_neighbors,
                    algorithm=self.algorithm,
                    leaf_size=self.leaf_size,
                    metric=self.metric,
                    p=self.p,
                    metric_params=self.metric_params,
                    n_jobs=self.n_jobs,
                )
                self._knn.fit(X)

                if self.contamination != "auto":
                    if not (0.0 < self.contamination <= 0.5):
                        raise ValueError(
                            "contamination must be in (0, 0.5], "
                            "got: %f" % self.contamination
                        )

                n_samples = self._knn.n_samples_fit_

                if self.n_neighbors > n_samples:
                    warnings.warn(
                        "n_neighbors (%s) is greater than the "
                        "total number of samples (%s). n_neighbors "
                        "will be set to (n_samples - 1) for estimation."
                        % (self.n_neighbors, n_samples)
                    )
                self.n_neighbors_ = max(1, min(self.n_neighbors, n_samples - 1))

                self._distances_fit_X_, _neighbors_indices_fit_X_ = self._knn.kneighbors(
                    n_neighbors=self.n_neighbors_
                )

                self._lrd = self._local_reachability_density(
                    self._distances_fit_X_, _neighbors_indices_fit_X_
                )

                # Compute lof score over training samples to define offset_:
                lrd_ratios_array = (
                    self._lrd[_neighbors_indices_fit_X_] / self._lrd[:, np.newaxis]
                )

                self.negative_outlier_factor_ = -np.mean(lrd_ratios_array, axis=1)

                if self.contamination == "auto":
                    # inliers score around -1 (the higher, the less abnormal).
                    self.offset_ = -1.5
                else:
                    self.offset_ = np.percentile(
                        self.negative_outlier_factor_, 100.0 * self.contamination
                    )

                for knn_prop_name in self._knn.__dict__.keys():
                    if knn_prop_name not in self.__dict__.keys():
                        setattr(self, knn_prop_name, self._knn.__dict__[knn_prop_name])

                return self

        def fit(self, X, y=None):
            return dispatch(
                self,
                "neighbors.LocalOutlierFactor.fit",
                {
                    "onedal": self.__class__._fit,
                    "sklearn": None,
                },
                X,
                y,
            )

        def _onedal_predict(self, X, queue=None):
            with config_context(target_offload=queue):
                check_is_fitted(self)

                if X is not None:
                    X = check_array(X, accept_sparse="csr")
                    is_inlier = np.ones(X.shape[0], dtype=int)
                    is_inlier[self.decision_function(X) < 0] = -1
                else:
                    is_inlier = np.ones(self._knn.n_samples_fit_, dtype=int)
                    is_inlier[self.negative_outlier_factor_ < self.offset_] = -1

                return is_inlier

        @wrap_output_data
        def _predict(self, X=None):
            return dispatch(
                self,
                "neighbors.LocalOutlierFactor.predict",
                {
                    "onedal": self.__class__._onedal_predict,
                    "sklearn": None,
                },
                X,
            )

        def _score_samples(self, X, queue=None):
            with config_context(target_offload=queue):
                check_is_fitted(self)
                X = check_array(X, accept_sparse="csr")

                distances_X, neighbors_indices_X = self._knn.kneighbors(
                    X, n_neighbors=self.n_neighbors_
                )
                X_lrd = self._local_reachability_density(distances_X, neighbors_indices_X)

                lrd_ratios_array = self._lrd[neighbors_indices_X] / X_lrd[:, np.newaxis]

                # as bigger is better:
                return -np.mean(lrd_ratios_array, axis=1)

        def _check_novelty_score_samples(self):
            if not self.novelty:
                msg = (
                    "score_samples is not available when novelty=False. The "
                    "scores of the training samples are always available "
                    "through the negative_outlier_factor_ attribute. Use "
                    "novelty=True if you want to use LOF for novelty detection "
                    "and compute score_samples for new unseen data."
                )
                raise AttributeError(msg)
            return True

        @available_if(_check_novelty_score_samples)
        @wrap_output_data
        def score_samples(self, X):
            return dispatch(
                self,
                "neighbors.LocalOutlierFactor.score_samples",
                {
                    "onedal": self.__class__._score_samples,
                    "sklearn": None,
                },
                X,
            )

        def _check_novelty_fit_predict(self):
            if self.novelty:
                msg = (
                    "fit_predict is not available when novelty=True. Use "
                    "novelty=False if you want to predict on the training set."
                )
                raise AttributeError(msg)
            return True

        def _fit_predict(self, X, y, queue=None):
            with config_context(target_offload=queue):
                return self.fit(X)._predict()

        @available_if(_check_novelty_fit_predict)
        @wrap_output_data
        def fit_predict(self, X, y=None):
            return dispatch(
                self,
                "neighbors.LocalOutlierFactor.fit_predict",
                {
                    "onedal": self.__class__._fit_predict,
                    "sklearn": None,
                },
                X,
                y,
            )

        def _onedal_gpu_supported(self, method_name, *data):
            class_name = self.__class__.__name__
            patching_status = PatchingConditionsChain(
                f"sklearn.neighbors.{class_name}.{method_name}"
            )
            return patching_status

        def _onedal_cpu_supported(self, method_name, *data):
            class_name = self.__class__.__name__
            patching_status = PatchingConditionsChain(
                f"sklearn.neighbors.{class_name}.{method_name}"
            )
            return patching_status

else:

    class LocalOutlierFactor(sklearn_LocalOutlierFactor):
        def __init__(
            self,
            n_neighbors=20,
            *,
            algorithm="auto",
            leaf_size=30,
            metric="minkowski",
            p=2,
            metric_params=None,
            contamination="auto",
            novelty=False,
            n_jobs=None,
        ):
            super().__init__(
                n_neighbors=n_neighbors,
                algorithm=algorithm,
                leaf_size=leaf_size,
                metric=metric,
                p=p,
                metric_params=metric_params,
                n_jobs=n_jobs,
                contamination=contamination,
                novelty=novelty,
            )

        def _fit(self, X, y=None, queue=None):
            with config_context(target_offload=queue):
                self._knn = NearestNeighbors(
                    n_neighbors=self.n_neighbors,
                    algorithm=self.algorithm,
                    leaf_size=self.leaf_size,
                    metric=self.metric,
                    p=self.p,
                    metric_params=self.metric_params,
                    n_jobs=self.n_jobs,
                )
                self._knn.fit(X)

                if self.contamination != "auto":
                    if not (0.0 < self.contamination <= 0.5):
                        raise ValueError(
                            "contamination must be in (0, 0.5], "
                            "got: %f" % self.contamination
                        )

                n_samples = self._knn.n_samples_fit_

                if self.n_neighbors > n_samples:
                    warnings.warn(
                        "n_neighbors (%s) is greater than the "
                        "total number of samples (%s). n_neighbors "
                        "will be set to (n_samples - 1) for estimation."
                        % (self.n_neighbors, n_samples)
                    )
                self.n_neighbors_ = max(1, min(self.n_neighbors, n_samples - 1))

                self._distances_fit_X_, _neighbors_indices_fit_X_ = self._knn.kneighbors(
                    n_neighbors=self.n_neighbors_
                )

                self._lrd = self._local_reachability_density(
                    self._distances_fit_X_, _neighbors_indices_fit_X_
                )

                # Compute lof score over training samples to define offset_:
                lrd_ratios_array = (
                    self._lrd[_neighbors_indices_fit_X_] / self._lrd[:, np.newaxis]
                )

                self.negative_outlier_factor_ = -np.mean(lrd_ratios_array, axis=1)

                if self.contamination == "auto":
                    # inliers score around -1 (the higher, the less abnormal).
                    self.offset_ = -1.5
                else:
                    self.offset_ = np.percentile(
                        self.negative_outlier_factor_, 100.0 * self.contamination
                    )

                for knn_prop_name in self._knn.__dict__.keys():
                    if knn_prop_name not in self.__dict__.keys():
                        setattr(self, knn_prop_name, self._knn.__dict__[knn_prop_name])

                return self

        def fit(self, X, y=None):
            return dispatch(
                self,
                "neighbors.LocalOutlierFactor.fit",
                {
                    "onedal": self.__class__._fit,
                    "sklearn": None,
                },
                X,
                y,
            )

        def _onedal_predict(self, X, queue=None):
            with config_context(target_offload=queue):
                check_is_fitted(self)

                if X is not None:
                    X = check_array(X, accept_sparse="csr")
                    is_inlier = np.ones(X.shape[0], dtype=int)
                    is_inlier[self.decision_function(X) < 0] = -1
                else:
                    is_inlier = np.ones(self._knn.n_samples_fit_, dtype=int)
                    is_inlier[self.negative_outlier_factor_ < self.offset_] = -1

                return is_inlier

        @wrap_output_data
        def _predict(self, X=None):
            return dispatch(
                self,
                "neighbors.LocalOutlierFactor.predict",
                {
                    "onedal": self.__class__._onedal_predict,
                    "sklearn": None,
                },
                X,
            )

        def _onedal_score_samples(self, X, queue=None):
            with config_context(target_offload=queue):
                check_is_fitted(self)
                X = check_array(X, accept_sparse="csr")

                distances_X, neighbors_indices_X = self._knn.kneighbors(
                    X, n_neighbors=self.n_neighbors_
                )
                X_lrd = self._local_reachability_density(distances_X, neighbors_indices_X)

                lrd_ratios_array = self._lrd[neighbors_indices_X] / X_lrd[:, np.newaxis]

                # as bigger is better:
                return -np.mean(lrd_ratios_array, axis=1)

        @wrap_output_data
        def _score_samples(self, X):
            if not self.novelty:
                msg = (
                    "score_samples is not available when novelty=False. The "
                    "scores of the training samples are always available "
                    "through the negative_outlier_factor_ attribute. Use "
                    "novelty=True if you want to use LOF for novelty detection "
                    "and compute score_samples for new unseen data."
                )
                raise AttributeError(msg)

            return dispatch(
                self,
                "neighbors.LocalOutlierFactor.score_samples",
                {
                    "onedal": self.__class__._onedal_score_samples,
                    "sklearn": None,
                },
                X,
            )

        def _onedal_fit_predict(self, X, y, queue=None):
            with config_context(target_offload=queue):
                return self.fit(X)._predict()

        @wrap_output_data
        def _fit_predict(self, X, y=None):
            return dispatch(
                self,
                "neighbors.LocalOutlierFactor._onedal_fit_predict",
                {
                    "onedal": self.__class__._onedal_fit_predict,
                    "sklearn": None,
                },
                X,
                y,
            )

        def _onedal_gpu_supported(self, method_name, *data):
            return True

        def _onedal_cpu_supported(self, method_name, *data):
            return True
