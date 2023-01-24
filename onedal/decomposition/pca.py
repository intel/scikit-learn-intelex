from onedal import _backend

from ..common._policy import _get_policy
from ..datatypes._data_conversion import from_table, to_table

import numpy as np


class PCA():
    def __init__(
        self,
        n_components=None,
        is_deterministic=False,
        method='precomputed',
        copy=True
    ):
        self.n_components = n_components
        self.copy = copy
        self.method = method
        self.is_deterministic = is_deterministic

    def get_onedal_params(self, data):
        return {
            'fptype':
                'float' if data.dtype is np.dtype('float32') else 'double',
            'method': self.method,
            'n_components': self.n_components,
            'is_deterministic': self.is_deterministic
        }

    def fit(self, X, y, queue):
        n_samples, n_features = X.shape
        n_sf_min = min(n_samples, n_features)

        policy = _get_policy(queue, X, y)
        params = self.get_onedal_params(X)
        cov_result = _backend.covariance.compute(
            policy,
            {'fptype': params['fptype'], 'method': 'dense'},
            to_table(X)
        )
        covariance_matrix = from_table(cov_result.cov_matrix)
        result = _backend.decomposition.dim_reduction.train(
            policy,
            params,
            to_table(covariance_matrix)
        )

        self.variances_ = from_table(result.variances)
        eigenvectors = from_table(result.eigenvectors)
        eigenvalues = from_table(result.eigenvalues)

        self.components_ = eigenvectors
        self.explained_variance_ = eigenvalues
        tot_var = self.explained_variance_.sum()
        self.explained_variance_ratio_ = self.explained_variance_ / tot_var
        self.singular_values_ = np.sqrt(
            (n_samples - 1) * self.explained_variance_
        )
        self.mean_ = from_table(result.means)
        self.n_components_ = self.n_components
        self.n_features_ = n_features
        # n_features_ was deprecated in scikit-learn 1.2 and
        # will be replaced by n_features_in_ in 1.4
        self.n_features_in_ = n_features
        self.n_samples_ = n_samples

        if self.n_components < n_sf_min:
            if self.explained_variance_.shape[0] == n_sf_min:
                self.noise_variance_ = \
                    self.explained_variance_[self.n_components:].mean()
            else:
                resid_var_ = self.variances_.sum()
                resid_var_ -= self.explained_variance_[:self.n_components].sum()
                self.noise_variance_ = \
                    resid_var_ / (n_sf_min - self.n_components)
        else:
            self.noise_variance_ = 0.

        #TODO: find out what should we do with this attribute
        self.feature_names_in_ = "Not supported yet"

        return self

    def _create_model(self):
        m = _backend.decomposition.dim_reduction.model()
        m.eigenvectors = to_table(self.components_)
        return m

    def predict(self, X, queue):
        policy = _get_policy(queue, X)
        params = self.get_onedal_params(X)
        model = self._create_model()
        result = _backend.decomposition.dim_reduction.infer(policy,
                                                            params,
                                                            model,
                                                            to_table(X))
        return from_table(result.transformed_data)
