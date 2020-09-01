#
#*******************************************************************************
# Copyright 2014-2020 Intel Corporation
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
#******************************************************************************/

import numpy as np

import sys
if sys.version_info[0] == 2:
    from sklearn.externals.six import string_types
else:
    string_types = str
import numbers
import warnings

import daal4py
from .._utils import (make2d, getFPType)

from sklearn.tree import (DecisionTreeClassifier, DecisionTreeRegressor)
from sklearn.tree._tree import (DTYPE, Tree)
from sklearn.ensemble import RandomForestClassifier as RandomForestClassifier_original
from sklearn.ensemble import RandomForestRegressor as RandomForestRegressor_original
from sklearn.utils import (check_random_state, check_array)
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import (check_is_fitted, check_consistent_length)
#from sklearn.ensemble._forest import _get_n_samples_bootstrap
from sklearn.base import clone
from sklearn.exceptions import DataConversionWarning, NotFittedError

from sklearn import __version__ as sklearn_version
from distutils.version import LooseVersion
from math import ceil
from scipy import sparse as sp

daal_version = tuple(map(int, (daal4py.__daal_link_version__[0:4], daal4py.__daal_link_version__[4:8])))


def _to_absolute_max_features(max_features, n_features, is_classification=False):
    if max_features is None:
        return n_features
    elif isinstance(max_features, string_types):
        if max_features == "auto":
            return max(1, int(np.sqrt(n_features))) if is_classification else n_features
        elif max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        elif max_features == "log2":
            return max(1, int(np.log2(n_features)))
        else:
            raise ValueError(
                'Invalid value for max_features. Allowed string '
                'values are "auto", "sqrt" or "log2".')
    elif isinstance(max_features, (numbers.Integral, np.integer)):
        return max_features
    else: # float
        if max_features > 0.0:
            return max(1, int(max_features * n_features))
        else:
            return 0

def _get_n_samples_bootstrap(n_samples, max_samples):
    if max_samples is None:
        return 1.

    if isinstance(max_samples, numbers.Integral):
        if not (1 <= max_samples <= n_samples):
            msg = "`max_samples` must be in range 1 to {} but got value {}"
            raise ValueError(msg.format(n_samples, max_samples))
        return float(max_samples/n_samples)

    if isinstance(max_samples, numbers.Real):
        if not (0 < max_samples < 1):
            msg = "`max_samples` must be in range (0, 1) but got value {}"
            raise ValueError(msg.format(max_samples))
        return float(max_samples)

    msg = "`max_samples` should be int or float, but got type '{}'"
    raise TypeError(msg.format(type(max_samples)))

def _daal_fit_classifier(self, X, y):
    #_check_daal_supported_parameters_classifier(self)
#    _supported_dtypes_ = [np.single, np.double]
#    X = check_array(X, dtype=_supported_dtypes_)
#    y = np.asarray(y)
#    y = np.atleast_1d(y)
#
#    if y.ndim == 2 and y.shape[1] == 1:
#        warnings.warn("A column-vector y was passed when a 1d array was"
#             " expected. Please change the shape of y to "
#             "(n_samples,), for example using ravel().",
#             DataConversionWarning, stacklevel=2)
#
#    check_consistent_length(X, y)
#
#    if y.ndim == 1:
#        # reshape is necessary to preserve the data contiguity against vs
#        # [:, np.newaxis] that does not.
#        y = np.reshape(y, (-1, 1))
#
#    self.n_outputs_ = y.shape[1]
#
#    if self.n_outputs_ != 1:
#        _class_name = self.__class__.__name__
#        raise ValueError(_class_name + " does not currently support multi-output data. Consider using OneHotEncoder")

    y = check_array(y, ensure_2d=False, dtype=None)
    y, _ = self._validate_y_class_weight(y)
    n_classes_ = self.n_classes_[0]
    classes_ = self.classes_[0]

    self.n_features_ = X.shape[1]

    rs_ = check_random_state(self.random_state)
    seed_ = rs_.randint(0, np.iinfo('i').max)

    if n_classes_ < 2:
        raise ValueError("Training data only contain information about one class.")

    # create algorithm
    X_fptype = getFPType(X)
    daal_engine_ = daal4py.engines_mt2203(seed=seed_, fptype=X_fptype)
    _featuresPerNode = _to_absolute_max_features(self.max_features, X.shape[1], is_classification=True)

    n_samples_bootstrap = _get_n_samples_bootstrap(
        n_samples=X.shape[0],
        max_samples=self.max_samples
    )

    if not self.bootstrap and self.oob_score:
        raise ValueError("Out of bag estimation only available"
                         " if bootstrap=True")
    
    print(f"self.min_samples_leaf = {self.min_samples_leaf}")

    dfc_algorithm = daal4py.decision_forest_classification_training(
        nClasses = int(n_classes_),
        fptype = X_fptype,
        method = 'defaultDense',
        nTrees = int(self.n_estimators),
        observationsPerTreeFraction = n_samples_bootstrap if self.bootstrap == True else 1.,
        featuresPerNode = int(_featuresPerNode),
        maxTreeDepth = int(0 if self.max_depth is None else self.max_depth),
        minObservationsInLeafNode = (self.min_samples_leaf if isinstance(self.min_samples_leaf, numbers.Integral)
                                     else int(ceil(self.min_samples_leaf * X.shape[0]))),
        engine = daal_engine_,
        impurityThreshold = float(0.0 if self.min_impurity_split is None else self.min_impurity_split),
        varImportance = "MDI",
        resultsToCompute = "",
        memorySavingMode = False,
        bootstrap = bool(self.bootstrap),
        minObservationsInSplitNode = (self.min_samples_split if isinstance(self.min_samples_split, numbers.Integral)
                                      else int(ceil(self.min_samples_split * X.shape[0]))),
        minWeightFractionInLeafNode = self.min_weight_fraction_leaf,
        minImpurityDecreaseInSplitNode = self.min_impurity_decrease,
        maxLeafNodes = 0 if self.max_leaf_nodes is None else self.max_leaf_nodes
    )
    self._cached_estimators_ = None
    # compute
    dfc_trainingResult = dfc_algorithm.compute(X, y)

    # get resulting model
    model = dfc_trainingResult.model
    self.daal_model_ = model

    # compute oob_score_
    if self.oob_score:
        self.estimators_ = self._estimators_
        self._set_oob_score(X, y)

    return self

def _daal_predict_classifier(self, X):
    X = self._validate_X_predict(X)

    if daal_version < (2020,1):
        dfc_algorithm = daal4py.decision_forest_classification_prediction(
            nClasses = int(self.n_classes_),
            fptype = 'float'
            )
    else:
        dfc_algorithm = daal4py.decision_forest_classification_prediction(
            nClasses = int(self.n_classes_),
            fptype = 'float',
            resultsToEvaluate="computeClassLabels"
            )
    dfc_predictionResult = dfc_algorithm.compute(X, self.daal_model_)

    pred = dfc_predictionResult.prediction

    return np.take(self.classes_, pred.ravel().astype(np.int, casting='unsafe'))

def _daal_predict_proba(self, X):
    X = self._validate_X_predict(X)

    dfc_algorithm = daal4py.decision_forest_classification_prediction(
        nClasses = int(self.n_classes_),
        fptype = 'float',
        resultsToEvaluate="computeClassProbabilities"
        )
    dfc_predictionResult = dfc_algorithm.compute(X, self.daal_model_)

    pred = dfc_predictionResult.probabilities

    return pred

def _fit_classifier(self, X, y, sample_weight=None):
    print("CLASSIFIER FIT")
    if sp.issparse(y):
        raise ValueError(
            "sparse multilabel-indicator for y is not supported."
        )
    _check_parameters(self, is_classification=True)

    daal_ready = (self.warm_start == False
        and self.criterion == "gini"
        and self.ccp_alpha == 0.0
        and sample_weight is None
        and self.class_weight is None and not sp.issparse(X))

    if daal_ready:
        _supported_dtypes_ = [np.single, np.double]
        X = check_array(X, dtype=_supported_dtypes_)
        y = np.asarray(y)
        y = np.atleast_1d(y)

        if y.ndim == 2 and y.shape[1] == 1:
            warnings.warn("A column-vector y was passed when a 1d array was"
                " expected. Please change the shape of y to "
                "(n_samples,), for example using ravel().",
                DataConversionWarning, stacklevel=2)

        check_consistent_length(X, y)

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]
        if self.n_outputs_ != 1:
            daal_ready = False

    if daal_ready:
        print(" DAAL CLASSIFIER FIT")
        _daal_fit_classifier(self, X, y)

        if not hasattr(self, "estimators_"):
            print("IN not HAS ATTR")
            self.estimators_ = self._estimators_
        
        # Decapsulate classes_ attributes
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            print("Decapsulate n_classes")
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]
        return self

    else:
        print("SKLEARN CLASSIFIER FIT")
        return super(RandomForestClassifier, self).fit(X, y, sample_weight=sample_weight)

#def _predict_classifier(self, X):
#    print("DAAL CLASSIFIER PREDICT")
#
#    return super(RandomForestClassifier, self).predict(X)

def _check_parameters(self, is_classification=False):
        #_class_name = self.__class__.__name__
        #if self.n_jobs != 1:
        #    warnings.warn(_class_name + ' ignores non-default settings of n_jobs')
        #if self.verbose != 0:
        #    warnings.wanr(_class_name + ' ignores non-default settings of verbose')
        #if self.warm_start:
        #    warnings.warn(_class_name + ' ignores non-default settings of warm_start')
        #if self.criterion != "gini":
        #    warnings.warn(_class_name + ' currently only supports criterion="gini"')
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
                raise ValueError("max_leaf_nodes must be integral number but was "
                                 "%r" % self.max_leaf_nodes)
            if self.max_leaf_nodes < 2:
                raise ValueError(("max_leaf_nodes {0} must be either None "
                                  "or larger than 1").format(self.max_leaf_nodes))

def _fit_regressor(self, X, y, sample_weight=None):
    print("DAAL REGRESSOR FIT")
    return super(RandomForestRegressor, self).fit(X, y, sample_weight=sample_weight)

def _predict_regressor(self, X):
    print("DAAL REGRESSOR PREDICT")
    return super(RandomForestRegressor, self).predict(X)

class RandomForestClassifier(RandomForestClassifier_original):
    __doc__ = RandomForestClassifier_original.__doc__

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
            max_samples=None):
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
        return _fit_classifier(self, X, y, sample_weight=sample_weight)


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

        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])

        if (not hasattr(self, 'daal_model_') or 
                sp.issparse(X) or self.n_outputs_ != 1 or
                not (X.dtype == np.float64 or X.dtype == np.float32)):
            return super(RandomForestClassifier, self).predict(X)
        else:
            return _daal_predict_classifier(self, X)

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

        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])

        if (not hasattr(self, 'daal_model_') or 
                sp.issparse(X) or self.n_outputs_ != 1 or
                not (X.dtype == np.float64 or X.dtype == np.float32)):
            return super(RandomForestClassifier, self).predict_proba(X)
        else:
            return _daal_predict_proba(self, X)

    @property
    def _estimators_(self):
        print("IN ESTIMATORS")
        if hasattr(self, '_cached_estimators_'):
            if self._cached_estimators_:
                return self._cached_estimators_
       
        if LooseVersion(sklearn_version) >= LooseVersion("0.22"):
            check_is_fitted(self)
        else:
            check_is_fitted(self, 'daal_model_')
        classes_ = self.classes_[0]
        n_classes_ = self.n_classes_[0]
        # convert model to estimators
        est = DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            min_impurity_split=self.min_impurity_split,
            random_state=None)
        # we need to set est.tree_ field with Trees constructed from Intel(R) DAAL solution
        estimators_ = []
        for i in range(self.n_estimators):
            # print("Tree #{}".format(i))
            est_i = clone(est)
            est_i.n_features_ = self.n_features_
            est_i.n_outputs_ = self.n_outputs_
            est_i.classes_ = classes_
            est_i.n_classes_ = n_classes_
            # treeState members: 'class_count', 'leaf_count', 'max_depth', 'node_ar', 'node_count', 'value_ar'
            tree_i_state_class = daal4py.getTreeState(self.daal_model_, i, n_classes_)
       
            node_ndarray = tree_i_state_class.node_ar
            value_ndarray = tree_i_state_class.value_ar
            value_shape = (node_ndarray.shape[0], self.n_outputs_,
                                 n_classes_)
            # assert np.allclose(value_ndarray, value_ndarray.astype(np.intc, casting='unsafe')), "Value array is non-integer"
            tree_i_state_dict = {
                'max_depth' : tree_i_state_class.max_depth,
                'node_count' : tree_i_state_class.node_count,
                'nodes' : tree_i_state_class.node_ar,
                'values': tree_i_state_class.value_ar }
            # 
            est_i.tree_ = Tree(self.n_features_, np.array([n_classes_], dtype=np.intp), self.n_outputs_)
            est_i.tree_.__setstate__(tree_i_state_dict)
            estimators_.append(est_i)
       
        self._cached_estimators_ = estimators_
        return estimators_
    

class RandomForestRegressor(RandomForestRegressor_original):
    __doc__ = RandomForestRegressor_original.__doc__

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
            max_samples=None):
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
        return _fit_regressor(self, X, y, sample_weight=sample_weight)


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

        return _predict_regressor(self, X)
