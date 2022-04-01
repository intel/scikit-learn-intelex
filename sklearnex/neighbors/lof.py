#!/usr/bin/env python
#===============================================================================
# Copyright 2022 Intel Corporation
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
#===============================================================================

import numpy as np
import warnings

from sklearn.neighbors._lof import LocalOutlierFactor as \
    sklearn_LocalOutlierFactor

from .knn_unsupervised import NearestNeighbors

try:
    from sklearn.utils.metaestimators import available_if
    from packaging.version import Version
except ImportError:
    from distutils.version import LooseVersion as Version

from sklearn import __version__ as sklearn_version

from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array

import logging
from .._utils import get_patch_message
from .._device_offload import dispatch, wrap_output_data
from .._config import config_context

if Version(sklearn_version) >= Version("1.0"):
    class LocalOutlierFactor(sklearn_LocalOutlierFactor, NearestNeighbors):
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
            NearestNeighbors.__init__(
                self,
                n_neighbors=n_neighbors,
                algorithm=algorithm,
                leaf_size=leaf_size,
                metric=metric,
                p=p,
                metric_params=metric_params,
                n_jobs=n_jobs,
            )

            self.contamination = contamination
            self.novelty = novelty

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

        def _fit(self, X, y, queue=None):
            with config_context(target_offload=queue):
                NearestNeighbors.fit(self, X)

                if self.contamination != "auto":
                    if not (0.0 < self.contamination <= 0.5):
                        raise ValueError(
                            "contamination must be in (0, 0.5], "
                            "got: %f" % self.contamination
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

                self._distances_fit_X_, _neighbors_indices_fit_X_ =\
                    NearestNeighbors.kneighbors(self, n_neighbors=self.n_neighbors_)

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

                return self

        def fit(self, X, y=None):
            """Fit the local outlier factor detector from the training dataset.
            Parameters
            ----------
            X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
                    (n_samples, n_samples) if metric='precomputed'
                Training data.
            y : Ignored
                Not used, present for API consistency by convention.
            Returns
            -------
            self : LocalOutlierFactor
                The fitted local outlier factor detector.
            """
            return dispatch(self, 'neighbors.LocalOutlierFactor.fit', {
                'onedal': self.__class__._fit,
                'sklearn': None,
            }, X, y)

        def _onedal_predict(self, X, queue=None):
            with config_context(target_offload=queue):
                check_is_fitted(self)

                if X is not None:
                    X = check_array(X, accept_sparse="csr")
                    is_inlier = np.ones(X.shape[0], dtype=int)
                    is_inlier[self.decision_function(X) < 0] = -1
                else:
                    is_inlier = np.ones(self.n_samples_fit_, dtype=int)
                    is_inlier[self.negative_outlier_factor_ < self.offset_] = -1

                return is_inlier

        @wrap_output_data
        def _predict(self, X=None):
            """Predict the labels (1 inlier, -1 outlier) of X according to LOF.
            If X is None, returns the same as fit_predict(X_train).
            Parameters
            ----------
            X : array-like of shape (n_samples, n_features), default=None
                The query sample or samples to compute the Local Outlier Factor
                w.r.t. to the training samples. If None, makes prediction on the
                training data without considering them as their own neighbors.
            Returns
            -------
            is_inlier : ndarray of shape (n_samples,)
                Returns -1 for anomalies/outliers and +1 for inliers.
            """
            return dispatch(self, 'neighbors.LocalOutlierFactor.predict', {
                'onedal': self.__class__._onedal_predict,
                'sklearn': None,
            }, X)

        def _score_samples(self, X, queue=None):
            with config_context(target_offload=queue):
                check_is_fitted(self)
                X = check_array(X, accept_sparse="csr")

                distances_X, neighbors_indices_X = NearestNeighbors.kneighbors(
                    self, X, n_neighbors=self.n_neighbors_
                )
                X_lrd = self._local_reachability_density(distances_X, neighbors_indices_X)

                lrd_ratios_array = self._lrd[neighbors_indices_X] / X_lrd[:, np.newaxis]

                # as bigger is better:
                return -np.mean(lrd_ratios_array, axis=1)

        @available_if(_check_novelty_score_samples)
        @wrap_output_data
        def score_samples(self, X):
            """Opposite of the Local Outlier Factor of X.
            It is the opposite as bigger is better, i.e. large values correspond
            to inliers.
            **Only available for novelty detection (when novelty is set to True).**
            The argument X is supposed to contain *new data*: if X contains a
            point from training, it considers the later in its own neighborhood.
            Also, the samples in X are not considered in the neighborhood of any
            point. Because of this, the scores obtained via ``score_samples`` may
            differ from the standard LOF scores.
            The standard LOF scores for the training data is available via the
            ``negative_outlier_factor_`` attribute.
            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                The query sample or samples to compute the Local Outlier Factor
                w.r.t. the training samples.
            Returns
            -------
            opposite_lof_scores : ndarray of shape (n_samples,)
                The opposite of the Local Outlier Factor of each input samples.
                The lower, the more abnormal.
            """
            return dispatch(self, 'neighbors.LocalOutlierFactor.score_samples', {
                'onedal': self.__class__._score_samples,
                'sklearn': None,
            }, X)

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
            """Fit the model to the training set X and return the labels.
            **Not available for novelty detection (when novelty is set to True).**
            Label is 1 for an inlier and -1 for an outlier according to the LOF
            score and the contamination parameter.
            Parameters
            ----------
            X : array-like of shape (n_samples, n_features), default=None
                The query sample or samples to compute the Local Outlier Factor
                w.r.t. to the training samples.
            y : Ignored
                Not used, present for API consistency by convention.
            Returns
            -------
            is_inlier : ndarray of shape (n_samples,)
                Returns -1 for anomalies/outliers and 1 for inliers.
            """

            # As fit_predict would be different from fit.predict, fit_predict is
            # only available for outlier detection (novelty=False)

            return dispatch(self, 'neighbors.LocalOutlierFactor.fit_predict', {
                'onedal': self.__class__._fit_predict,
                'sklearn': None,
            }, X, y)

        def _onedal_gpu_supported(self, method_name, *data):
            return True

        def _onedal_cpu_supported(self, method_name, *data):
            return True
else:
    class LocalOutlierFactor(sklearn_LocalOutlierFactor, NearestNeighbors):
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
            NearestNeighbors.__init__(
                self,
                n_neighbors=n_neighbors,
                algorithm=algorithm,
                leaf_size=leaf_size,
                metric=metric,
                p=p,
                metric_params=metric_params,
                n_jobs=n_jobs,
            )

            self.contamination = contamination
            self.novelty = novelty

        def _fit(self, X, y=None, queue=None):
            """Fit the local outlier factor detector from the training dataset.
            Parameters
            ----------
            X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
                    (n_samples, n_samples) if metric='precomputed'
                Training data.
            y : Ignored
                Not used, present for API consistency by convention.
            Returns
            -------
            self : LocalOutlierFactor
                The fitted local outlier factor detector.
            """

            with config_context(target_offload=queue):
                NearestNeighbors.fit(self, X)

                if self.contamination != "auto":
                    if not (0.0 < self.contamination <= 0.5):
                        raise ValueError(
                            "contamination must be in (0, 0.5], "
                            "got: %f" % self.contamination
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

                self._distances_fit_X_, _neighbors_indices_fit_X_ =\
                    NearestNeighbors.kneighbors(self, n_neighbors=self.n_neighbors_)

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

                return self

        def fit(self, X, y=None):
            """Fit the local outlier factor detector from the training dataset.
            Parameters
            ----------
            X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
                    (n_samples, n_samples) if metric='precomputed'
                Training data.
            y : Ignored
                Not used, present for API consistency by convention.
            Returns
            -------
            self : LocalOutlierFactor
                The fitted local outlier factor detector.
            """
            return dispatch(self, 'neighbors.LocalOutlierFactor.fit', {
                'onedal': self.__class__._fit,
                'sklearn': None,
            }, X, y)

        def _onedal_score_samples(self, X, queue=None):
            with config_context(target_offload=queue):
                check_is_fitted(self)
                X = check_array(X, accept_sparse="csr")

                distances_X, neighbors_indices_X = NearestNeighbors.kneighbors(
                    self, X, n_neighbors=self.n_neighbors_
                )
                X_lrd = self._local_reachability_density(distances_X, neighbors_indices_X)

                lrd_ratios_array = self._lrd[neighbors_indices_X] / X_lrd[:, np.newaxis]

                # as bigger is better:
                return -np.mean(lrd_ratios_array, axis=1)

        @wrap_output_data
        def score_samples(self, X):
            """Opposite of the Local Outlier Factor of X.
            It is the opposite as bigger is better, i.e. large values correspond
            to inliers.
            **Only available for novelty detection (when novelty is set to True).**
            The argument X is supposed to contain *new data*: if X contains a
            point from training, it considers the later in its own neighborhood.
            Also, the samples in X are not considered in the neighborhood of any
            point. Because of this, the scores obtained via ``score_samples`` may
            differ from the standard LOF scores.
            The standard LOF scores for the training data is available via the
            ``negative_outlier_factor_`` attribute.
            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                The query sample or samples to compute the Local Outlier Factor
                w.r.t. the training samples.
            Returns
            -------
            opposite_lof_scores : ndarray of shape (n_samples,)
                The opposite of the Local Outlier Factor of each input samples.
                The lower, the more abnormal.
            """
            return dispatch(self, 'neighbors.LocalOutlierFactor.score_samples', {
                'onedal': self.__class__._onedal_score_samples,
                'sklearn': None,
            }, X)

        def _onedal_predict(self, X, queue=None):
            with config_context(target_offload=queue):
                check_is_fitted(self)

                if X is not None:
                    X = check_array(X, accept_sparse="csr")
                    is_inlier = np.ones(X.shape[0], dtype=int)
                    is_inlier[self.decision_function(X) < 0] = -1
                else:
                    is_inlier = np.ones(self.n_samples_fit_, dtype=int)
                    is_inlier[self.negative_outlier_factor_ < self.offset_] = -1

                return is_inlier

        @wrap_output_data
        def _predict(self, X=None):
            """Predict the labels (1 inlier, -1 outlier) of X according to LOF.
            If X is None, returns the same as fit_predict(X_train).
            Parameters
            ----------
            X : array-like of shape (n_samples, n_features), default=None
                The query sample or samples to compute the Local Outlier Factor
                w.r.t. to the training samples. If None, makes prediction on the
                training data without considering them as their own neighbors.
            Returns
            -------
            is_inlier : ndarray of shape (n_samples,)
                Returns -1 for anomalies/outliers and +1 for inliers.
            """
            return dispatch(self, 'neighbors.LocalOutlierFactor.predict', {
                'onedal': self.__class__._onedal_predict,
                'sklearn': None,
            }, X)

        def _score_samples(self, X):
            """Opposite of the Local Outlier Factor of X.
            It is the opposite as bigger is better, i.e. large values correspond
            to inliers.
            **Only available for novelty detection (when novelty is set to True).**
            The argument X is supposed to contain *new data*: if X contains a
            point from training, it considers the later in its own neighborhood.
            Also, the samples in X are not considered in the neighborhood of any
            point. Because of this, the scores obtained via ``score_samples`` may
            differ from the standard LOF scores.
            The standard LOF scores for the training data is available via the
            ``negative_outlier_factor_`` attribute.
            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                The query sample or samples to compute the Local Outlier Factor
                w.r.t. the training samples.
            Returns
            -------
            opposite_lof_scores : ndarray of shape (n_samples,)
                The opposite of the Local Outlier Factor of each input samples.
                The lower, the more abnormal.
            """
            logging.info("sklearn.neighbors.LocalOutlierFactor."
                         "score_samples: " + get_patch_message("onedal"))
            check_is_fitted(self)
            X = check_array(X, accept_sparse="csr")

            distances_X, neighbors_indices_X = NearestNeighbors.kneighbors(
                self, X, n_neighbors=self.n_neighbors_
            )
            X_lrd = self._local_reachability_density(distances_X, neighbors_indices_X)

            lrd_ratios_array = self._lrd[neighbors_indices_X] / X_lrd[:, np.newaxis]

            # as bigger is better:
            return -np.mean(lrd_ratios_array, axis=1)

        def _onedal_fit_predict(self, X, y, queue=None):
            with config_context(target_offload=queue):
                return self.fit(X)._predict()

        @wrap_output_data
        def _fit_predict(self, X, y=None):
            """Fit the model to the training set X and return the labels.
            **Not available for novelty detection (when novelty is set to True).**
            Label is 1 for an inlier and -1 for an outlier according to the LOF
            score and the contamination parameter.
            Parameters
            ----------
            X : array-like of shape (n_samples, n_features), default=None
                The query sample or samples to compute the Local Outlier Factor
                w.r.t. to the training samples.
            y : Ignored
                Not used, present for API consistency by convention.
            Returns
            -------
            is_inlier : ndarray of shape (n_samples,)
                Returns -1 for anomalies/outliers and 1 for inliers.
            """

            # As fit_predict would be different from fit.predict, fit_predict is
            # only available for outlier detection (novelty=False)

            return dispatch(self, 'neighbors.LocalOutlierFactor._onedal_fit_predict', {
                'onedal': self.__class__._onedal_fit_predict,
                'sklearn': None,
            }, X, y)

        def _onedal_gpu_supported(self, method_name, *data):
            return True

        def _onedal_cpu_supported(self, method_name, *data):
            return True
