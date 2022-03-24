import numpy as np
import warnings

from .knn_unsupervised import NearestNeighbors

from sklearn.neighbors._lof import LocalOutlierFactor as \
    sklearn_LocalOutlierFactor

from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array

import logging
from .._utils import get_patch_message


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
        logging.info("sklearn.neighbors.LocalOutlierFactor."
                     "fit: " + get_patch_message("onedal"))
        NearestNeighbors.fit(self, X)

        if self.contamination != "auto":
            if not (0.0 < self.contamination <= 0.5):
                raise ValueError(
                    "contamination must be in (0, 0.5], got: %f" % self.contamination
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

        self._distances_fit_X_, _neighbors_indices_fit_X_ = NearestNeighbors.kneighbors(
            self, n_neighbors=self.n_neighbors_
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

        return self

    @available_if(_check_novelty_score_samples)
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
