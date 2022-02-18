#===============================================================================
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
#===============================================================================

from ._common import BaseSVR
from .._device_offload import dispatch, wrap_output_data

from sklearn.svm import SVR as sklearn_SVR
from sklearn.utils.validation import _deprecate_positional_args
from sklearn import __version__ as sklearn_version

try:
    from packaging.version import Version
except ImportError:
    from distutils.version import LooseVersion as Version
from onedal.svm import SVR as onedal_SVR


class SVR(sklearn_SVR, BaseSVR):
    __doc__ = sklearn_SVR.__doc__

    @_deprecate_positional_args
    def __init__(self, *, kernel='rbf', degree=3, gamma='scale',
                 coef0=0.0, tol=1e-3, C=1.0, epsilon=0.1, shrinking=True,
                 cache_size=200, verbose=False, max_iter=-1):
        super().__init__(
            kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, tol=tol, C=C,
            epsilon=epsilon, shrinking=shrinking, cache_size=cache_size, verbose=verbose,
            max_iter=max_iter)

    def fit(self, X, y, sample_weight=None):
        """
        Fit the SVM model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) \
                or (n_samples, n_samples)
            Training vectors, where `n_samples` is the number of samples
            and `n_features` is the number of features.
            For kernel="precomputed", the expected shape of X is
            (n_samples, n_samples).

        y : array-like of shape (n_samples,)
            Target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,), default=None
            Per-sample weights. Rescale C per sample. Higher weights
            force the classifier to put more emphasis on these points.

        Returns
        -------
        self : object
            Fitted estimator.

        Notes
        -----
        If X and y are not C-ordered and contiguous arrays of np.float64 and
        X is not a scipy.sparse.csr_matrix, X and/or y may be copied.

        If X is a dense array, then the other methods will not support sparse
        matrices as input.
        """
        if Version(sklearn_version) >= Version("1.0"):
            self._check_feature_names(X, reset=True)
        dispatch(self, 'svm.SVR.fit', {
            'onedal': self.__class__._onedal_fit,
            'sklearn': sklearn_SVR.fit,
        }, X, y, sample_weight)

        return self

    @wrap_output_data
    def predict(self, X):
        """
        Perform regression on samples in X.

        For an one-class model, +1 (inlier) or -1 (outlier) is returned.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            For kernel="precomputed", the expected shape of X is
            (n_samples_test, n_samples_train).

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted values.
        """
        if Version(sklearn_version) >= Version("1.0"):
            self._check_feature_names(X, reset=False)
        return dispatch(self, 'svm.SVR.predict', {
            'onedal': self.__class__._onedal_predict,
            'sklearn': sklearn_SVR.predict,
        }, X)

    def _onedal_gpu_supported(self, method_name, *data):
        return False

    def _onedal_cpu_supported(self, method_name, *data):
        if method_name == 'svm.SVR.fit':
            return self.kernel in ['linear', 'rbf', 'poly', 'sigmoid']
        if method_name == 'svm.SVR.predict':
            return hasattr(self, '_onedal_estimator')

    def _onedal_fit(self, X, y, sample_weight=None, queue=None):
        onedal_params = {
            'C': self.C,
            'epsilon': self.epsilon,
            'kernel': self.kernel,
            'degree': self.degree,
            'gamma': self.gamma,
            'coef0': self.coef0,
            'tol': self.tol,
            'shrinking': self.shrinking,
            'cache_size': self.cache_size,
            'max_iter': self.max_iter,
        }

        self._onedal_estimator = onedal_SVR(**onedal_params)
        self._onedal_estimator.fit(X, y, sample_weight, queue=queue)
        self._save_attributes()

    def _onedal_predict(self, X, queue=None):
        return self._onedal_estimator.predict(X, queue=queue)
