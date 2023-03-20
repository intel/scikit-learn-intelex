#===============================================================================
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
#===============================================================================

# from onedal.spmd.ensemble import RandomForestClassifier, RandomForestRegressor
from onedal.spmd.ensemble import RandomForestRegressor

# TODO:
# Currently it uses `onedal` module interface.
# Add sklearnex dispatching.

from daal4py.sklearn._utils import (
    daal_check_version, sklearn_check_version,
    make2d, get_dtype
)

import numpy as np

import numbers

from abc import ABC

from ..._device_offload import dispatch, wrap_output_data

from sklearnex.ensemble import RandomForestClassifier as sklearn_RandomForestClassifier
from sklearnex.ensemble import RandomForestRegressor as sklearn_RandomForestRegressor

from sklearn.utils.validation import (
    check_is_fitted,
    check_consistent_length,
    check_array,
    check_X_y)

from onedal.datatypes import _num_features, _num_samples

from sklearn.utils import check_random_state, deprecated

from sklearn.base import clone

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree._tree import Tree

from onedal.spmd.ensemble import RandomForestClassifier as onedal_RandomForestClassifier
from onedal.spmd.ensemble import RandomForestRegressor as onedal_RandomForestRegressor

from scipy import sparse as sp

if sklearn_check_version('1.2'):
    from sklearn.utils._param_validation import Interval

class BaseRandomForest(ABC):

    def _save_attributes(self):
        self._onedal_model = self._onedal_estimator._onedal_model
        # TODO:
        # update for regression
        if self.oob_score:
            self.oob_score_ = self._onedal_estimator.oob_score_
            self.oob_prediction_ = self._onedal_estimator.oob_prediction_
        return self

class RandomForestClassifier(sklearn_RandomForestClassifier, BaseRandomForest):
    __doc__ = sklearn_RandomForestClassifier.__doc__

    if sklearn_check_version('1.2'):
        _parameter_constraints: dict = {
            **sklearn_RandomForestClassifier._parameter_constraints,
            "max_bins": [Interval(numbers.Integral, 2, None, closed="left")],
            "min_bin_size": [Interval(numbers.Integral, 1, None, closed="left")]
        }

    if sklearn_check_version('1.0'):
        def __init__(
                self,
                n_estimators=100,
                criterion="gini",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.,
                max_features='sqrt' if sklearn_check_version('1.1') else 'auto',
                max_leaf_nodes=None,
                min_impurity_decrease=0.,
                bootstrap=True,
                oob_score=False,
                n_jobs=None,
                random_state=None,
                verbose=0,
                warm_start=False,
                class_weight=None,
                ccp_alpha=0.0,
                max_samples=None,
                max_bins=256,
                min_bin_size=1):
            super(RandomForestClassifier, self).__init__(
                n_estimators=n_estimators,
                criterion=criterion,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                min_impurity_decrease=min_impurity_decrease,
                bootstrap=bootstrap,
                oob_score=oob_score,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose,
                warm_start=warm_start,
                class_weight=class_weight
            )
            self.warm_start = warm_start
            self.ccp_alpha = ccp_alpha
            self.max_samples = max_samples
            self.max_bins = max_bins
            self.min_bin_size = min_bin_size
            self.min_impurity_split = None
            # self._estimator = DecisionTreeClassifier()
    else:
        def __init__(self,
                     n_estimators=100,
                     criterion="gini",
                     max_depth=None,
                     min_samples_split=2,
                     min_samples_leaf=1,
                     min_weight_fraction_leaf=0.,
                     max_features="auto",
                     max_leaf_nodes=None,
                     min_impurity_decrease=0.,
                     min_impurity_split=None,
                     bootstrap=True,
                     oob_score=False,
                     n_jobs=None,
                     random_state=None,
                     verbose=0,
                     warm_start=False,
                     class_weight=None,
                     ccp_alpha=0.0,
                     max_samples=None,
                     max_bins=256,
                     min_bin_size=1):
            super(RandomForestClassifier, self).__init__(
                n_estimators=n_estimators,
                criterion=criterion,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                min_impurity_decrease=min_impurity_decrease,
                min_impurity_split=min_impurity_split,
                bootstrap=bootstrap,
                oob_score=oob_score,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose,
                warm_start=warm_start,
                class_weight=class_weight,
                ccp_alpha=ccp_alpha,
                max_samples=max_samples
            )
            self.warm_start = warm_start
            self.ccp_alpha = ccp_alpha
            self.max_samples = max_samples
            self.max_bins = max_bins
            self.min_bin_size = min_bin_size
            self.min_impurity_split = None
            # self._estimator = DecisionTreeClassifier()

    def fit(self, X, y, sample_weight=None):
        """
        Build a forest of trees from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        Returns
        -------
        self : object
        """
        dispatch(self, 'ensemble.RandomForestClassifier.fit', {
            'onedal': self.__class__._onedal_fit,
            'sklearn': sklearn_RandomForestClassifier.fit,
        }, X, y, sample_weight)
        return self

    @wrap_output_data
    def predict(self, X):
        """
        Predict class for X.

        The predicted class of an input sample is a vote by the trees in
        the forest, weighted by their probability estimates. That is,
        the predicted class is the one with highest mean probability
        estimate across the trees.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted classes.
        """
        return dispatch(self, 'ensemble.RandomForestClassifier.predict', {
            'onedal': self.__class__._onedal_predict,
            'sklearn': sklearn_RandomForestClassifier.predict,
        }, X)

    @wrap_output_data
    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        The predicted class probabilities of an input sample are computed as
        the mean predicted class probabilities of the trees in the forest.
        The class probability of a single tree is the fraction of samples of
        the same class in a leaf.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes), or a list of n_outputs
            such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        if sklearn_check_version("1.0"):
            self._check_feature_names(X, reset=False)
        if hasattr(self, 'n_features_in_'):
            try:
                num_features = _num_features(X)
            except TypeError:
                num_features = _num_samples(X)
            if num_features != self.n_features_in_:
                raise ValueError(
                    (f'X has {num_features} features, '
                     f'but RandomForestClassifier is expecting '
                     f'{self.n_features_in_} features as input'))
        return dispatch(self, 'ensemble.RandomForestClassifier.predict_proba', {
            'onedal': self.__class__._onedal_predict_proba,
            'sklearn': sklearn_RandomForestClassifier.predict_proba,
        }, X)

    if sklearn_check_version('1.0'):
        @deprecated(
            "Attribute `n_features_` was deprecated in version 1.0 and will be "
            "removed in 1.2. Use `n_features_in_` instead.")
        @property
        def n_features_(self):
            return self.n_features_in_

    def _onedal_cpu_supported(self, method_name, *data):
        if method_name == 'ensemble.RandomForestClassifier.fit':
            ready, X, y, sample_weight = self._onedal_ready(*data)
            if not ready:
                return False
            elif sp.issparse(X):
                return False
            elif sp.issparse(y):
                return False
            elif sp.issparse(sample_weight):
                return False
            elif not self.ccp_alpha == 0.0:
                return False
            elif self.warm_start:
                return False
            elif self.oob_score and not daal_check_version((2023, 'P', 101)):
                return False
            elif not self.n_outputs_ == 1:
                return False
            elif hasattr(self, 'estimators_'):
                return False
            else:
                return True
        if method_name in ['ensemble.RandomForestClassifier.predict',
                           'ensemble.RandomForestClassifier.predict_proba']:
            X = data[0]
            if not hasattr(self, '_onedal_model'):
                return False
            elif sp.issparse(X):
                return False
            elif not (hasattr(self, 'n_outputs_') and self.n_outputs_ == 1):
                return False
            elif not daal_check_version((2021, 'P', 400)):
                return False
            elif self.warm_start:
                return False
            else:
                return True
        raise RuntimeError(
            f'Unknown method {method_name} in {self.__class__.__name__}')

    def _onedal_gpu_supported(self, method_name, *data):
        if method_name == 'ensemble.RandomForestClassifier.fit':
            # TODO:
            # raise error that is not supported.
            #ready, X, y, sample_weight = self._onedal_ready(*data)
            #if not ready:
            #    return False
            #elif sp.issparse(X):
            #    return False
            #elif sp.issparse(y):
            #    return False
            #elif sp.issparse(sample_weight):
            #    return False
            #elif not sample_weight:  # `sample_weight` is not supported.
            #    return False
            #elif not self.ccp_alpha == 0.0:
            #    return False
            #elif self.warm_start:
            #    return False
            #elif self.oob_score:
            #    return False
            #elif not self.n_outputs_ == 1:
            #    return False
            #elif hasattr(self, 'estimators_'):
            #    return False
            #else:
            return True
        if method_name in ['ensemble.RandomForestClassifier.predict',
                           'ensemble.RandomForestClassifier.predict_proba']:
            # X = data[0]
            # if not hasattr(self, '_onedal_model'):
            #     return False
            # elif sp.issparse(X):
            #     return False
            # elif not (hasattr(self, 'n_outputs_') and self.n_outputs_ == 1):
            #     return False
            # elif not daal_check_version((2021, 'P', 400)):
            #     return False
            # elif self.warm_start:
            #     return False
            # else:
            return True
        raise RuntimeError(
            f'Unknown method {method_name} in {self.__class__.__name__}')

    def _onedal_fit(self, X, y, sample_weight=None, queue=None):
        if self.oob_score:
            err = 'out_of_bag_error_accuracy|out_of_bag_error_decision_function'
        else:
            err = 'none'

        onedal_params = {
            'n_estimators': self.n_estimators,
            'criterion': self.criterion,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'min_weight_fraction_leaf': self.min_weight_fraction_leaf,
            'max_features': self.max_features,
            'max_leaf_nodes': self.max_leaf_nodes,
            'min_impurity_decrease': self.min_impurity_decrease,
            'min_impurity_split': self.min_impurity_split,
            'bootstrap': self.bootstrap,
            'oob_score': self.oob_score,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'warm_start': self.warm_start,
            'error_metric_mode': err,
            'variable_importance_mode': 'mdi',
            'class_weight': self.class_weight,
            'max_bins': self.max_bins,
            'min_bin_size': self.min_bin_size,
            'max_samples': self.max_samples
        }
        self._cached_estimators_ = None

        # Compute
        self._onedal_estimator = onedal_RandomForestClassifier(**onedal_params)
        self._onedal_estimator.fit(X, y, sample_weight, queue=queue)

        self._save_attributes()
        if sklearn_check_version("1.2"):
            self._estimator = DecisionTreeClassifier()
        return self

    def _onedal_predict(self, X, queue=None):
        X = check_array(X, dtype=[np.float32, np.float64])
        check_is_fitted(self)
        if sklearn_check_version("1.0"):
            self._check_feature_names(X, reset=False)

        res = self._onedal_estimator.predict(X, queue=queue)
        return res.ravel().astype(np.int64, casting='unsafe')

    def _onedal_predict_proba(self, X, queue=None):
        X = check_array(X, dtype=[np.float64, np.float32])
        check_is_fitted(self)
        if sklearn_check_version('0.23'):
            self._check_n_features(X, reset=False)
        if sklearn_check_version("1.0"):
            self._check_feature_names(X, reset=False)
        return self._onedal_estimator.predict_proba(X, queue=queue)
