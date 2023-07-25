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

from daal4py.sklearn._utils import sklearn_check_version
from ._common import BaseSVC
from .._device_offload import dispatch, wrap_output_data

from sklearn.svm import NuSVC as sklearn_NuSVC
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.exceptions import NotFittedError
if sklearn_check_version('1.0'):
    from sklearn.utils.metaestimators import available_if

from onedal.svm import NuSVC as onedal_NuSVC


class NuSVC(sklearn_NuSVC, BaseSVC):
    __doc__ = sklearn_NuSVC.__doc__

    if sklearn_check_version('1.2'):
        _parameter_constraints: dict = {**sklearn_NuSVC._parameter_constraints}

    @_deprecate_positional_args
    def __init__(self, *, nu=0.5, kernel='rbf', degree=3, gamma='scale',
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=200, class_weight=None,
                 verbose=False, max_iter=-1, decision_function_shape='ovr',
                 break_ties=False, random_state=None):
        super().__init__(
            nu=nu, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0,
            shrinking=shrinking, probability=probability, tol=tol, cache_size=cache_size,
            class_weight=class_weight, verbose=verbose, max_iter=max_iter,
            decision_function_shape=decision_function_shape, break_ties=break_ties,
            random_state=random_state)

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
        if sklearn_check_version("1.2"):
            self._validate_params()
        if sklearn_check_version("1.0"):
            self._check_feature_names(X, reset=True)
        dispatch(self, 'fit', {
            'onedal': self.__class__._onedal_fit,
            'sklearn': sklearn_NuSVC.fit,
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
        if sklearn_check_version("1.0"):
            self._check_feature_names(X, reset=False)
        return dispatch(self, 'predict', {
            'onedal': self.__class__._onedal_predict,
            'sklearn': sklearn_NuSVC.predict,
        }, X)

    if sklearn_check_version('1.0'):
        @available_if(sklearn_NuSVC._check_proba)
        def predict_proba(self, X):
            """
            Compute probabilities of possible outcomes for samples in X.

            The model need to have probability information computed at training
            time: fit with attribute `probability` set to True.

            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                For kernel="precomputed", the expected shape of X is
                (n_samples_test, n_samples_train).

            Returns
            -------
            T : ndarray of shape (n_samples, n_classes)
                Returns the probability of the sample for each class in
                the model. The columns correspond to the classes in sorted
                order, as they appear in the attribute :term:`classes_`.

            Notes
            -----
            The probability model is created using cross validation, so
            the results can be slightly different than those obtained by
            predict. Also, it will produce meaningless results on very small
            datasets.
            """
            return self._predict_proba(X)
    else:
        @property
        def predict_proba(self):
            self._check_proba()
            return self._predict_proba

    @wrap_output_data
    def _predict_proba(self, X):
        if sklearn_check_version("1.0"):
            self._check_feature_names(X, reset=False)
        sklearn_pred_proba = (sklearn_NuSVC.predict_proba
                              if sklearn_check_version("1.0")
                              else sklearn_NuSVC._predict_proba)

        return dispatch(self, 'predict_proba', {
            'onedal': self.__class__._onedal_predict_proba,
            'sklearn': sklearn_pred_proba,
        }, X)

    @wrap_output_data
    def decision_function(self, X):
        if sklearn_check_version("1.0"):
            self._check_feature_names(X, reset=False)
        return dispatch(self, 'decision_function', {
            'onedal': self.__class__._onedal_decision_function,
            'sklearn': sklearn_NuSVC.decision_function,
        }, X)

    def _onedal_fit(self, X, y, sample_weight=None, queue=None):
        onedal_params = {
            'nu': self.nu,
            'kernel': self.kernel,
            'degree': self.degree,
            'gamma': self.gamma,
            'coef0': self.coef0,
            'tol': self.tol,
            'shrinking': self.shrinking,
            'cache_size': self.cache_size,
            'max_iter': self.max_iter,
            'class_weight': self.class_weight,
            'break_ties': self.break_ties,
            'decision_function_shape': self.decision_function_shape,
        }

        self._onedal_estimator = onedal_NuSVC(**onedal_params)
        self._onedal_estimator.fit(X, y, sample_weight, queue=queue)

        if self.probability:
            self._fit_proba(X, y, sample_weight, queue=queue)
        self._save_attributes()

    def _onedal_predict(self, X, queue=None):
        return self._onedal_estimator.predict(X, queue=queue)

    def _onedal_predict_proba(self, X, queue=None):
        if getattr(self, 'clf_prob', None) is None:
            raise NotFittedError(
                "predict_proba is not available when fitted with probability=False")
        from .._config import get_config, config_context

        # We use stock metaestimators below, so the only way
        # to pass a queue is using config_context.
        cfg = get_config()
        cfg['target_offload'] = queue
        with config_context(**cfg):
            return self.clf_prob.predict_proba(X)

    def _onedal_decision_function(self, X, queue=None):
        return self._onedal_estimator.decision_function(X, queue=queue)
