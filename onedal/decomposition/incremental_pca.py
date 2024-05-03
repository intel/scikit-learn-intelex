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

from daal4py.sklearn._utils import get_dtype

from ..datatypes import _convert_to_supported, from_table, to_table
from ..utils import _check_array
from .pca import BasePCA


class IncrementalPCA(BasePCA):
    """
    Incremental estimator for PCA based on oneDAL implementation.
    Allows to compute PCA if data are splitted into batches.

    Parameters
    ----------
    n_components : int, default=None
        Number of components to keep. If ``n_components`` is ``None``,
        then ``n_components`` is set to ``min(n_samples, n_features)``.

    is_deterministic : bool, default=True
        When True the ``components_`` vectors are chosen in deterministic
        way, otherwise some of them can be oppositely directed.

    method : string, default='cov'
        Method used on oneDAL side to compute result.

    whiten : bool, default=False
        When True the ``components_`` vectors are divided
        by ``n_samples`` times ``components_`` to ensure uncorrelated outputs
        with unit component-wise variances.

        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometimes
        improve the predictive accuracy of the downstream estimators by
        making data respect some hard-wired assumptions.

    Attributes
    ----------
        components_ : ndarray of shape (n_components, n_features)
        Principal axes in feature space, representing the directions of
        maximum variance in the data. Equivalently, the right singular
        vectors of the centered input data, parallel to its eigenvectors.
        The components are sorted by decreasing ``explained_variance_``.

        explained_variance_ : ndarray of shape (n_components,)
            Variance explained by each of the selected components.

        explained_variance_ratio_ : ndarray of shape (n_components,)
            Percentage of variance explained by each of the selected components.
            If all components are stored, the sum of explained variances is equal
            to 1.0.

        singular_values_ : ndarray of shape (n_components,)
            The singular values corresponding to each of the selected components.
            The singular values are equal to the 2-norms of the ``n_components``
            variables in the lower-dimensional space.

        mean_ : ndarray of shape (n_features,)
            Per-feature empirical mean, aggregate over calls to ``partial_fit``.

        var_ : ndarray of shape (n_features,)
            Per-feature empirical variance, aggregate over calls to
            ``partial_fit``.

        noise_variance_ : float
            Equal to the average of (min(n_features, n_samples) - n_components)
            smallest eigenvalues of the covariance matrix of X.

    """

    def __init__(
        self,
        n_components=None,
        is_deterministic=True,
        method="cov",
        whiten=False,
    ):
        self.n_components = n_components
        self.method = method
        self.is_deterministic = is_deterministic
        self.whiten = whiten
        module = self._get_backend("decomposition", "dim_reduction")
        self._partial_result = module.partial_train_result()

    def _reset(self):
        module = self._get_backend("decomposition", "dim_reduction")
        del self.components_
        self._partial_result = module.partial_train_result()

    def partial_fit(self, X, queue):
        """Incremental fit with X. All of X is processed as a single batch.

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
        X = _check_array(X)
        n_samples, n_features = X.shape

        first_pass = not hasattr(self, "components_")
        if first_pass:
            self.components_ = None
            self.n_samples_seen_ = n_samples
            self.n_features_in_ = n_features
        else:
            self.n_samples_seen_ += n_samples

        if self.n_components is None:
            if self.components_ is None:
                self.n_components_ = min(n_samples, n_features)
            else:
                self.n_components_ = self.components_.shape[0]
        else:
            self.n_components_ = self.n_components

        module = self._get_backend("decomposition", "dim_reduction")

        if not hasattr(self, "_policy"):
            self._policy = self._get_policy(queue, X)

        X = _convert_to_supported(self._policy, X)

        if not hasattr(self, "_dtype"):
            self._dtype = get_dtype(X)
            self._params = self._get_onedal_params(X)

        X_table = to_table(X)
        self._partial_result = module.partial_train(
            self._policy, self._params, self._partial_result, X_table
        )
        return self

    def finalize_fit(self, queue=None):
        """
        Finalizes principal components computation and obtains resulting
        attributes from the current `_partial_result`.

        Parameters
        ----------
        queue : dpctl.SyclQueue
            Not used here, added for API conformance

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        module = self._get_backend("decomposition", "dim_reduction")
        result = module.finalize_train(self._policy, self._params, self._partial_result)
        self.mean_ = from_table(result.means).ravel()
        self.var_ = from_table(result.variances).ravel()
        self.components_ = from_table(result.eigenvectors)
        self.singular_values_ = np.nan_to_num(from_table(result.singular_values).ravel())
        self.explained_variance_ = np.maximum(from_table(result.eigenvalues).ravel(), 0)
        self.explained_variance_ratio_ = from_table(
            result.explained_variances_ratio
        ).ravel()
        self.noise_variance_ = self._compute_noise_variance(
            self.n_components_, min(self.n_samples_seen_, self.n_features_in_)
        )

        return self
