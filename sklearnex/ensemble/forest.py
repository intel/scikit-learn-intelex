# BUILD: 0023
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

from daal4py.sklearn._utils import (
    daal_check_version, sklearn_check_version,
)
import numpy as np

import numbers
import warnings

from abc import ABC
import warnings
from sklearn.exceptions import DataConversionWarning

from .._device_offload import dispatch, wrap_output_data

from sklearn.ensemble import RandomForestClassifier as sklearn_RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor as sklearn_RandomForestRegressor

from sklearn.utils.validation import (
    check_is_fitted,
    check_consistent_length,
    check_array,
    _num_samples)

from sklearn.utils import (check_random_state, check_array, deprecated)

from sklearn.base import clone

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree._tree import Tree

from onedal.ensemble import RandomForestClassifier as onedal_RandomForestClassifier
from onedal.ensemble import RandomForestRegressor as onedal_RandomForestRegressor
from onedal.primitives import get_tree_state_cls, get_tree_state_reg
from onedal.datatypes import _check_X_y

from scipy import sparse as sp


class BaseRandomForest(ABC):
    def _fit_proba(self, X, y, sample_weight=None, queue=None):
        from .._config import get_config, config_context

        params = self.get_params()
        clf_base = self.__class__(**params)

        # We use stock metaestimators below, so the only way
        # to pass a queue is using config_context.
        cfg = get_config()
        cfg['target_offload'] = queue

    def _save_attributes(self):
        self._onedal_model = self._onedal_estimator._onedal_model
        # TODO:
        # update for regression
        if self.oob_score:
            self.oob_score_ = self._onedal_estimator.oob_score_
        return self

    # TODO:
    # move to onedal modul.
    def _check_parameters(self):
        if not self.bootstrap and self.max_samples is not None:
            raise ValueError(
                "`max_sample` cannot be set if `bootstrap=False`. "
                "Either switch to `bootstrap=True` or set "
                "`max_sample=None`."
            )
        if isinstance(self.min_samples_leaf, numbers.Integral):
            if not 1 <= self.min_samples_leaf:
                raise ValueError("min_samples_leaf must be at least 1 "
                                 "or in (0, 0.5], got %s"
                                 % self.min_samples_leaf)
        else:  # float
            if not 0. < self.min_samples_leaf <= 0.5:
                raise ValueError("min_samples_leaf must be at least 1 "
                                 "or in (0, 0.5], got %s"
                                 % self.min_samples_leaf)
        if isinstance(self.min_samples_split, numbers.Integral):
            if not 2 <= self.min_samples_split:
                raise ValueError("min_samples_split must be an integer "
                                 "greater than 1 or a float in (0.0, 1.0]; "
                                 "got the integer %s"
                                 % self.min_samples_split)
        else:  # float
            if not 0. < self.min_samples_split <= 1.:
                raise ValueError("min_samples_split must be an integer "
                                 "greater than 1 or a float in (0.0, 1.0]; "
                                 "got the float %s"
                                 % self.min_samples_split)
        if not 0 <= self.min_weight_fraction_leaf <= 0.5:
            raise ValueError("min_weight_fraction_leaf must in [0, 0.5]")
        if self.min_impurity_split is not None:
            warnings.warn("The min_impurity_split parameter is deprecated. "
                          "Its default value has changed from 1e-7 to 0 in "
                          "version 0.23, and it will be removed in 0.25. "
                          "Use the min_impurity_decrease parameter instead.",
                          FutureWarning)

            if self.min_impurity_split < 0.:
                raise ValueError("min_impurity_split must be greater than "
                                 "or equal to 0")
        if self.min_impurity_decrease < 0.:
            raise ValueError("min_impurity_decrease must be greater than "
                             "or equal to 0")
        if self.max_leaf_nodes is not None:
            if not isinstance(self.max_leaf_nodes, numbers.Integral):
                raise ValueError(
                    "max_leaf_nodes must be integral number but was "
                    "%r" %
                    self.max_leaf_nodes)
            if self.max_leaf_nodes < 2:
                raise ValueError(
                    ("max_leaf_nodes {0} must be either None "
                     "or larger than 1").format(
                        self.max_leaf_nodes))
        if isinstance(self.max_bins, numbers.Integral):
            if not 2 <= self.max_bins:
                raise ValueError("max_bins must be at least 2, got %s"
                                 % self.max_bins)
        else:
            raise ValueError("max_bins must be integral number but was "
                             "%r" % self.max_bins)
        if isinstance(self.min_bin_size, numbers.Integral):
            if not 1 <= self.min_bin_size:
                raise ValueError("min_bin_size must be at least 1, got %s"
                                 % self.min_bin_size)
        else:
            raise ValueError("min_bin_size must be integral number but was "
                             "%r" % self.min_bin_size)

    def check_sample_weight(sample_weight, X, dtype=None):
        n_samples = _num_samples(X)

        if dtype is not None and dtype not in [np.float32, np.float64]:
            dtype = np.float64

        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=dtype)
        elif isinstance(sample_weight, numbers.Number):
            sample_weight = np.full(n_samples, sample_weight, dtype=dtype)
        else:
            if dtype is None:
                dtype = [np.float64, np.float32]
            sample_weight = check_array(
                sample_weight,
                accept_sparse=False,
                ensure_2d=False,
                dtype=dtype,
                order="C")
            if sample_weight.ndim != 1:
                raise ValueError("Sample weights must be 1D array or scalar")

            if sample_weight.shape != (n_samples,):
                raise ValueError("sample_weight.shape == {}, expected {}!"
                                 .format(sample_weight.shape, (n_samples,)))
        return sample_weight

    def _onedal_fit_data_supported(self, X, y, sample_weight):
        try:
            X_, y_ = _check_X_y(
                X, y, dtype=[np.float64, np.float32],
                 force_all_finite=True, accept_sparse=False)
            y_ = np.asarray(y_)
            if y_.ndim == 1:
                # reshape is necessary to preserve the data contiguity against vs
                # [:, np.newaxis] that does not.
                y_ = np.reshape(y_, (-1, 1))
            n_outputs_ = y_.shape[1]
            if n_outputs_ != 1:
                raise ValueError(
                    "Number of outputs ({n_outputs_}) is not 1."
                )
            if sample_weight is not None:
                sample_weight_ = self.check_sample_weight(sample_weight, X)
            self._fit_data_support = True
            self.n_outputs_ = n_outputs_
            return X_, y_, sample_weight_
        except:
            self._fit_data_support = False
            return X, y, sample_weight

    def _onedal_fit_supported(self, X, y, sample_weight):
            # FIXME:
            # should be aligned with old backend lib interface.
            #if not (self.oob_score and daal_check_version(
            #        (2021, 'P', 500)) or not self.oob_score):
            #    pass
            if self.oob_score:
                pass
            elif not (self.criterion == "gini"):
                pass
            elif not (self.ccp_alpha == 0.0):
                pass
            elif self.warm_start is not False:
                pass
            else:
                X_, y_, sample_weight_ = self._onedal_fit_data_supported(X, y, sample_weight)
                if self._fit_data_support:
                    return X_, y_, sample_weight_
                else:
                    return X, y, sample_weight
            self._fit_data_support = False
            return X, y, sample_weight


class RandomForestClassifier(sklearn_RandomForestClassifier, BaseRandomForest):
    __doc__ = sklearn_RandomForestClassifier.__doc__

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
            self.max_bins = max_bins
            self.min_bin_size = min_bin_size
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
        if not self.bootstrap and self.max_samples is not None:
            raise ValueError(
                "`max_sample` cannot be set if `bootstrap=False`. "
                "Either switch to `bootstrap=True` or set "
                "`max_sample=None`."
            )
        X, y, sample_weight = self._onedal_fit_supported(X, y, sample_weight)
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
        # TODO:
        # _check_proba()
        # self._check_proba()
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

    @property
    def _estimators_(self):
        if hasattr(self, '_cached_estimators_'):
            if self._cached_estimators_:
                return self._cached_estimators_
        if sklearn_check_version('0.22'):
            check_is_fitted(self)
        else:
            check_is_fitted(self, '_onedal_model')
        classes_ = self.classes_[0]
        n_classes_ = self.n_classes_[0]
        # convert model to estimators
        params = {
            'criterion': self.criterion,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'min_weight_fraction_leaf': self.min_weight_fraction_leaf,
            'max_features': self.max_features,
            'max_leaf_nodes': self.max_leaf_nodes,
            'min_impurity_decrease': self.min_impurity_decrease,
            'random_state': None,
        }
        if not sklearn_check_version('1.0'):
            params['min_impurity_split'] = self.min_impurity_split
        est = DecisionTreeClassifier(**params)
        # TODO:
        # we need to set est.tree_ field with Trees constructed from Intel(R)
        # oneAPI Data Analytics Library solution
        estimators_ = []
        random_state_checked = check_random_state(self.random_state)
        for i in range(self.n_estimators):
            est_i = clone(est)
            est_i.set_params(
                random_state=random_state_checked.randint(
                    np.iinfo(
                        np.int32).max))
            if sklearn_check_version('1.0'):
                est_i.n_features_in_ = self.n_features_in_
            else:
                est_i.n_features_ = self.n_features_in_
            est_i.n_outputs_ = self.n_outputs_
            est_i.classes_ = classes_
            est_i.n_classes_ = n_classes_
            # treeState members: 'class_count', 'leaf_count', 'max_depth',
            # 'node_ar', 'node_count', 'value_ar'
            tree_i_state_class = get_tree_state_cls(
                self._onedal_model, i, n_classes_)
            # node_ndarray = tree_i_state_class.node_ar
            # value_ndarray = tree_i_state_class.value_ar
            # value_shape = (node_ndarray.shape[0], self.n_outputs_,
            #                n_classes_)
            # assert np.allclose(
            #     value_ndarray, value_ndarray.astype(np.intc, casting='unsafe')
            # ), "Value array is non-integer"
            tree_i_state_dict = {
                'max_depth': tree_i_state_class.max_depth,
                'node_count': tree_i_state_class.node_count,
                'nodes': tree_i_state_class.node_ar,
                'values': tree_i_state_class.value_ar}
            est_i.tree_ = Tree(
                self.n_features_in_,
                np.array(
                    [n_classes_],
                    dtype=np.intp),
                self.n_outputs_)
            est_i.tree_.__setstate__(tree_i_state_dict)
            estimators_.append(est_i)
        self._cached_estimators_ = estimators_
        return estimators_

    def _onedal_cpu_supported(self, method_name, *data):
        if method_name == 'ensemble.RandomForestClassifier.fit':
            return self._fit_data_support
        if method_name in ['ensemble.RandomForestClassifier.predict',
                           'ensemble.RandomForestClassifier.predict_proba']:
            if not hasattr(self, '_onedal_model'):
                pass
            elif sp.issparse(data[0]):
                pass
            elif not (hasattr(self, 'n_outputs_') and self.n_outputs_ == 1):
                pass
            elif not daal_check_version((2021, 'P', 400)):
                pass
            else:
                return True
            return False
        raise RuntimeError(
            f'Unknown method {method_name} in {self.__class__.__name__}')

    def _onedal_gpu_supported(self, method_name, *data):
        if method_name == 'ensemble.RandomForestClassifier.fit':
            # `sample_weight` is not supported.
            return len(data) == 2 and self._fit_data_support
        if method_name in ['ensemble.RandomForestClassifier.predict',
                           'ensemble.RandomForestClassifier.predict_proba']:
            if not hasattr(self, '_onedal_model'):
                pass
            elif sp.issparse(data[0]):
                pass
            elif not (hasattr(self, 'n_outputs_') and self.n_outputs_ == 1):
                pass
            elif not daal_check_version((2021, 'P', 400)):
                pass
            else:
                return True
            return False
        raise RuntimeError(
            f'Unknown method {method_name} in {self.__class__.__name__}')

    def _onedal_fit(self, X, y, sample_weight=None, queue=None):
        if sp.issparse(y):
            raise ValueError(
                "sparse multilabel-indicator for y is not supported."
            )
        if sklearn_check_version("1.2"):
            self._validate_params()
        else:
            self._check_parameters()

        if sklearn_check_version("1.0"):
            self._check_feature_names(X, reset=True)
        X = check_array(X, dtype=[np.float32, np.float64])
        y = np.asarray(y)
        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warnings.warn("A column-vector y was passed when a 1d array was"
                          " expected. Please change the shape of y to "
                          "(n_samples,), for example using ravel().",
                          DataConversionWarning, stacklevel=2)
        check_consistent_length(X, y)

        y = check_array(y, ensure_2d=False, dtype=None)
        y, expanded_class_weight = self._validate_y_class_weight(y)
        n_classes_ = self.n_classes_[0]
        self.n_features_in_ = X.shape[1]
        if not sklearn_check_version('1.0'):
            self.n_features_ = self.n_features_in_

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight
        if sample_weight is not None:
            sample_weight = [sample_weight]

        rs_ = check_random_state(self.random_state)
        seed_ = rs_.randint(0, np.iinfo('i').max)

        if n_classes_ < 2:
            raise ValueError(
                "Training data only contain information about one class.")

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
            'bootstrap': self.bootstrap,
            'oob_score': self.oob_score,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'warm_start': self.warm_start,
            'error_metric_mode' : 'out_of_bag_error' if self.oob_score else 'none',
            'variable_importance_mode' : 'mdi',
            'class_weight': self.class_weight,
        }
        self._cached_estimators_ = None
        # Compute
        self._onedal_estimator = onedal_RandomForestClassifier(**onedal_params)
        self._onedal_estimator.classes_ = self.classes_
        self._onedal_estimator.fit(X, y, sample_weight, queue=queue)

        self._save_attributes()
        if sklearn_check_version("1.2"):
            self._estimator = DecisionTreeClassifier()
        self.estimators_ = self._estimators_
        # Decapsulate classes_ attributes
        self.n_classes_ = self.n_classes_[0]
        self.classes_ = self.classes_[0]
        return self

    def _onedal_predict(self, X, queue=None):
        if sklearn_check_version("1.0"):
            self._check_feature_names(X, reset=False)
        X = check_array(
            X,
            accept_sparse=False,  # is not supported
            dtype=[np.float64, np.float32]
        )
        return self._onedal_estimator.predict(X, queue=queue)

    def _onedal_predict_proba(self, X, queue=None):
        if sklearn_check_version("1.0"):
            self._check_feature_names(X, reset=False)
        X = check_array(
            X,
            accept_sparse=False,  # is not supported
            dtype=[np.float64, np.float32]
        )
        return self._onedal_estimator.predict_proba(X, queue=queue)


class RandomForestRegressor(sklearn_RandomForestRegressor, BaseRandomForest):
    __doc__ = sklearn_RandomForestRegressor.__doc__

    if sklearn_check_version('1.0'):
        def __init__(
                self,
                n_estimators=100,
                *,
                criterion="squared_error",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.,
                max_features=1.0 if sklearn_check_version('1.1') else 'auto',
                max_leaf_nodes=None,
                min_impurity_decrease=0.,
                bootstrap=True,
                oob_score=False,
                n_jobs=None,
                random_state=None,
                verbose=0,
                warm_start=False,
                ccp_alpha=0.0,
                max_samples=None,
                max_bins=256,
                min_bin_size=1):
            super(RandomForestRegressor, self).__init__(
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
                warm_start=warm_start
            )
            self.ccp_alpha = ccp_alpha
            self.max_samples = max_samples
            self.max_bins = max_bins
            self.min_bin_size = min_bin_size
            self.min_impurity_split = None
    else:
        def __init__(self,
                     n_estimators=100, *,
                     criterion="mse",
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
                     ccp_alpha=0.0,
                     max_samples=None,
                     max_bins=256,
                     min_bin_size=1):
            super(RandomForestRegressor, self).__init__(
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
                ccp_alpha=ccp_alpha,
                max_samples=max_samples
            )
            self.ccp_alpha = ccp_alpha
            self.max_samples = max_samples
            self.max_bins = max_bins
            self.min_bin_size = min_bin_size
            self.min_impurity_split = None

    @property
    def _estimators_(self):
        if hasattr(self, '_cached_estimators_'):
            if self._cached_estimators_:
                return self._cached_estimators_
        if sklearn_check_version('0.22'):
            check_is_fitted(self)
        else:
            check_is_fitted(self, '_onedal_model')
        # convert model to estimators
        params = {
            'criterion': self.criterion,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'min_weight_fraction_leaf': self.min_weight_fraction_leaf,
            'max_features': self.max_features,
            'max_leaf_nodes': self.max_leaf_nodes,
            'min_impurity_decrease': self.min_impurity_decrease,
            'random_state': None,
        }
        if not sklearn_check_version('1.0'):
            params['min_impurity_split'] = self.min_impurity_split
        est = DecisionTreeRegressor(**params)
        # we need to set est.tree_ field with Trees constructed from Intel(R)
        # oneAPI Data Analytics Library solution
        estimators_ = []
        random_state_checked = check_random_state(self.random_state)
        for i in range(self.n_estimators):
            est_i = clone(est)
            est_i.set_params(
                random_state=random_state_checked.randint(
                    np.iinfo(
                        np.int32).max))
            if sklearn_check_version('1.0'):
                est_i.n_features_in_ = self.n_features_in_
            else:
                est_i.n_features_ = self.n_features_in_
            est_i.n_outputs_ = self.n_outputs_
            tree_i_state_class = get_tree_state_reg(
                self._onedal_model, i)
            tree_i_state_dict = {
                'max_depth': tree_i_state_class.max_depth,
                'node_count': tree_i_state_class.node_count,
                'nodes': tree_i_state_class.node_ar,
                'values': tree_i_state_class.value_ar}

            est_i.tree_ = Tree(
                self.n_features_in_, np.array(
                    [1], dtype=np.intp), self.n_outputs_)
            est_i.tree_.__setstate__(tree_i_state_dict)
            estimators_.append(est_i)
        # self._cached_estimators_ = estimators_
        return estimators_

    def _onedal_cpu_supported(self, method_name, *data):
        if method_name == 'ensemble.RandomForestRegressor.fit':
            return self._fit_data_support
        if method_name == 'ensemble.RandomForestRegressor.predict':
            if not hasattr(self, '_onedal_model'):
                pass
            elif sp.issparse(data[0]):
                pass
            elif not (hasattr(self, 'n_outputs_') and self.n_outputs_ == 1):
                pass
            else:
                return True
            return False
        raise RuntimeError(
            f'Unknown method {method_name} in {self.__class__.__name__}')

    def _onedal_gpu_supported(self, method_name, *data):
        if method_name == 'ensemble.RandomForestClassifier.fit':
            # `sample_weight` is not supported.
            return len(data) == 2 and self._fit_data_support
        if method_name == 'ensemble.RandomForestRegressor.predict':
            if not hasattr(self, '_onedal_model'):
                pass
            elif sp.issparse(data[0]):
                pass
            elif not (hasattr(self, 'n_outputs_') and self.n_outputs_ == 1):
                pass
            else:
                return True
            return False
        raise RuntimeError(
            f'Unknown method {method_name} in {self.__class__.__name__}')

    def _onedal_fit(self, X, y, sample_weight=None, queue=None):
        if sp.issparse(y):
            raise ValueError(
                "sparse multilabel-indicator for y is not supported."
            )
        if sklearn_check_version("1.2"):
            self._validate_params()
        else:
            self._check_parameters()
        if sklearn_check_version("1.0"):
            self._check_feature_names(X, reset=True)
        X = check_array(X, dtype=[np.float64, np.float32])
        y = np.asarray(y)
        y = np.atleast_1d(y)
        y = check_array(y, ensure_2d=False, dtype=X.dtype)
        check_consistent_length(X, y)
        self.n_features_in_ = X.shape[1]
        if not sklearn_check_version('1.0'):
            self.n_features_ = self.n_features_in_
        rs_ = check_random_state(self.random_state)
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
            'bootstrap': self.bootstrap,
            'oob_score': self.oob_score,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'warm_start': self.warm_start,
            'error_metric_mode' : 'out_of_bag_error' if self.oob_score else 'none',
            'variable_importance_mode' : 'mdi'
        }
        self._cached_estimators_ = None
        self._onedal_estimator = onedal_RandomForestRegressor(**onedal_params)
        self._onedal_estimator.fit(X, y, sample_weight, queue=queue)

        self._save_attributes()
        if sklearn_check_version("1.2"):
            self._estimator = DecisionTreeRegressor()
        self.estimators_ = self._estimators_
        return self

    def _onedal_predict(self, X, queue=None):
        if sklearn_check_version("1.0"):
            self._check_feature_names(X, reset=False)
        X = check_array(
            X,
            accept_sparse=False,
            dtype=[np.float64, np.float32]
        )
        return self._onedal_estimator.predict(X, queue=queue)

    @wrap_output_data
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
        if not self.bootstrap and self.max_samples is not None:
            raise ValueError(
                "`max_sample` cannot be set if `bootstrap=False`. "
                "Either switch to `bootstrap=True` or set "
                "`max_sample=None`."
            )
        X, y, sample_weight = self._onedal_fit_supported(X, y, sample_weight)
        dispatch(self, 'ensemble.RandomForestRegressor.fit', {
            'onedal': self.__class__._onedal_fit,
            'sklearn': sklearn_RandomForestRegressor.fit,
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
        return dispatch(self, 'ensemble.RandomForestRegressor.predict', {
            'onedal': self.__class__._onedal_predict,
            'sklearn': sklearn_RandomForestRegressor.predict,
        }, X)

    if sklearn_check_version('1.0'):
        @deprecated(
            "Attribute `n_features_` was deprecated in version 1.0 and will be "
            "removed in 1.2. Use `n_features_in_` instead.")
        @property
        def n_features_(self):
            return self.n_features_in_
