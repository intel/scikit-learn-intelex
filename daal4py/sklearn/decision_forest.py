#
#*******************************************************************************
# Copyright 2014-2017 Intel Corporation
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
from sklearn.externals.six import string_types
import numbers
import warnings

import daal4py
from .utils import (make2d, getFPType)

from sklearn.tree import (DecisionTreeClassifier, DecisionTreeRegressor)
from sklearn.tree._tree import (DTYPE, Tree)
from sklearn.ensemble import RandomForestClassifier as skl_RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor as skl_RandomForestRegressor
from sklearn.utils import (check_random_state, check_array)
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import (check_is_fitted, check_consistent_length)
from sklearn.base import clone
from sklearn.exceptions import DataConversionWarning, NotFittedError

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


class RandomForestClassifier(skl_RandomForestClassifier):
    def __init__(self,
            n_estimators=10,
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
            n_jobs=1,
            random_state=None,
            verbose=0,
            warm_start=False):
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
                warm_start=warm_start
            )


    def _check_daal_supported_parameters(self):
        _class_name = self.__class__.__name__
        print(_class_name)
        if self.n_jobs != 1:
            warnings.warn(_class_name + ' ignores non-default settings of n_jobs')
        if self.verbose != 0:
            warnings.wanr(_class_name + ' ignores non-default settings of verbose')
        if self.warm_start:
            warnings.warn(_class_name + ' ignores non-default settings of warm_start')
        if self.criterion != "gini":
            warnings.warn(_class_name + ' currently only supports criterion="gini"')
        if self.min_impurity_decrease != 0.0:
            warnings.warn(_class_name + " currently does not support min_impurity_decrease."
                          "It currently supports min_impurity_split to control tree growth.")
        if self.max_leaf_nodes is not None:
            warnings.warn(_class_name + " currently does not support non-default "
                          "setting for max_leaf_nodes.")
        if self.min_weight_fraction_leaf != 0.0:
            warnings.warn(_class_name + " currently does not support non-default "
                          "setting for min_weight_fraction_leaf.")
        if self.min_samples_leaf != 1:
            warnings.warn(_class_name + " currently does not support non-default "
                          "setting for min_samples_leaf.")


    def daal_fit(self, X, y):
        self._check_daal_supported_parameters()
        _supported_dtypes_ = [np.single, np.double]
        X = check_array(X, dtype=_supported_dtypes_)
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
            _class_name = self.__class__.__name__
            raise ValueError(_class_name + " does not currently support multi-output data. Consider using OneHotEncoder")

        y = check_array(y, ensure_2d=False, dtype=None)
        y, _ = self._validate_y_class_weight(y)
        self.n_classes_ = self.n_classes_[0]
        self.classes_ = self.classes_[0]

        self.n_features_ = X.shape[1]

        rs_ = check_random_state(self.random_state)
        seed_ = rs_.randint(0, np.iinfo('i').max)

        if self.n_classes_ < 2:
            raise ValueError("Training data only contain information about one class.")

        # create algorithm
        X_fptype = getFPType(X)
        daal_engine_ = daal4py.engines_mt2203(seed=seed_, fptype=X_fptype)
        _featuresPerNode = _to_absolute_max_features(self.max_features, X.shape[1], is_classification=False)

        dfc_algorithm = daal4py.decision_forest_classification_training(
            nClasses=int(self.n_classes_),
            fptype=X_fptype,
            method='defaultDense',
            nTrees=int(self.n_estimators),
            observationsPerTreeFraction=1,
            featuresPerNode=float(_featuresPerNode),
            maxTreeDepth=int(0 if self.max_depth is None else self.max_depth),
            minObservationsInLeafNode=1,
            engine=daal_engine_,
            impurityThreshold=float(0.0 if self.min_impurity_split is None else self.min_impurity_split),
            varImportance="MDI",
            resultsToCompute="",
            memorySavingMode=False,
            bootstrap=bool(self.bootstrap)
        )
        # compute
        dfc_trainingResult = dfc_algorithm.compute(X, y)

        # get resulting model
        model = dfc_trainingResult.model
        self.daal_model_ = model

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
        
        # we need to set est.tree_ field with Trees constructed from DAAL solution
        estimators_ = []
        for i in range(self.n_estimators):
            # print("Tree #{}".format(i))
            est_i = clone(est)
            est_i.n_features_ = self.n_features_
            est_i.n_outputs_ = self.n_outputs_
            est_i.classes_ = self.classes_
            est_i.n_classes_ = self.n_classes_
            # treeState members: 'class_count', 'leaf_count', 'max_depth', 'node_ar', 'node_count', 'value_ar'
            tree_i_state_class = daal4py.getTreeState(model, i, self.n_classes_)

            node_ndarray = tree_i_state_class.node_ar
            value_ndarray = tree_i_state_class.value_ar
            value_shape = (node_ndarray.shape[0], self.n_outputs_,
                                 self.n_classes_)

            assert np.allclose(value_ndarray, value_ndarray.astype(np.intc, casting='unsafe')), "Value array is non-integer"

            tree_i_state_dict = {
                'max_depth' : tree_i_state_class.max_depth,
                'node_count' : tree_i_state_class.node_count,
                'nodes' : tree_i_state_class.node_ar,
                'values': tree_i_state_class.value_ar }
            # 
            est_i.tree_ = Tree(self.n_features_, np.array([self.n_classes_], dtype=np.intp), self.n_outputs_)
            est_i.tree_.__setstate__(tree_i_state_dict)
            estimators_.append(est_i)

        self.estimators_ = estimators_

        # compute oob_score_
        if self.oob_score:
            self._set_oob_score(X, y)

        return self


    def daal_predict(self, X):
        X = self._validate_X_predict(X)

        dfc_algorithm = daal4py.decision_forest_classification_prediction(
            nClasses = int(self.n_classes_),
            fptype = 'float'
            )
        dfc_predictionResult = dfc_algorithm.compute(X, self.daal_model_)

        pred = dfc_predictionResult.prediction

        return np.take(self.classes_, pred.ravel().astype(np.int, casting='unsafe'))


    def fit(self, X, y):
        return self.daal_fit(X, y)


    def predict(self, X):
        check_is_fitted(self, 'daal_model_')
        return self.daal_predict(X)



class RandomForestRegressor(skl_RandomForestRegressor):
    def __init__(self,
            n_estimators=10,
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
            n_jobs=1,
            random_state=None,
            verbose=0,
            warm_start=False):
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
                warm_start=warm_start
            )
    
    def _check_daal_supported_parameters(self):
        _class_name = self.__class__.__name__
        if self.n_jobs != 1:
            warnings.warn(_class_name + ' ignores non-default settings of n_jobs')
        if self.verbose != 0:
            warnings.warn(_class_name + ' ignores non-default settings of verbose')
        if self.warm_start:
            warnings.warn(_class_name + ' ignores non-default settings of warm_start')
        if self.criterion != "mse":
            warnings.warn(_class_name + ' currently only supports criterion="mse"')
        if self.min_impurity_decrease != 0.0:
            warnings.warn(_class_name + " currently does not support min_impurity_decrease."
                          "It currently supports min_impurity_split to control tree growth.")
        if self.max_leaf_nodes is not None:
            warnings.warn(_class_name + " currently does not support non-default "
                          "setting for max_leaf_nodes.")
        if self.min_weight_fraction_leaf != 0.0:
            warnings.warn(_class_name + " currently does not support non-default "
                          "setting for min_weight_fraction_leaf.")
        if self.min_samples_leaf != 1:
            warnings.warn(_class_name + " currently does not support non-default "
                          "setting for min_samples_leaf.")


    # daal only supports "mse" criterion
    def daal_fit(self, X, y):
        self._check_daal_supported_parameters()
        _supported_dtypes_ = [np.double, np.single]
        X = check_array(X, dtype=_supported_dtypes_)
        y = np.atleast_1d(y)

        if y.ndim == 2 and y.shape[1] == 1:
            warnings.warn("A column-vector y was passed when a 1d array was"
                 " expected. Please change the shape of y to "
                 "(n_samples,), for example using ravel().",
                 DataConversionWarning, stacklevel=2)

        y = check_array(y, ensure_2d=False, dtype=X.dtype)
        check_consistent_length(X, y)

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]
        self.n_features_ = X.shape[1]
        rs_ = check_random_state(self.random_state)

        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available"
                             " if bootstrap=True")

        X_fptype = getFPType(X)
        seed_ = rs_.randint(0, np.iinfo('i').max)
        daal_engine = daal4py.engines_mt2203(seed=seed_, fptype=X_fptype)

        _featuresPerNode = _to_absolute_max_features(self.max_features, X.shape[1], is_classification=False)

        # create algorithm
        dfr_algorithm = daal4py.decision_forest_regression_training(
            fptype = getFPType(X),
            method='defaultDense',
            nTrees=int(self.n_estimators),
            observationsPerTreeFraction=1,
            featuresPerNode=float(_featuresPerNode),
            maxTreeDepth=int(0 if self.max_depth is None else self.max_depth),
            minObservationsInLeafNode=1,
            engine=daal_engine,
            impurityThreshold=float(0.0 if self.min_impurity_split is None else self.min_impurity_split),
            varImportance="MDI",
            resultsToCompute="",
            memorySavingMode=False,
            bootstrap=bool(self.bootstrap)
        )

        dfr_trainingResult = dfr_algorithm.compute(X, y)

        # get resulting model
        model = dfr_trainingResult.model
        self.daal_model_ = model

        # convert model to estimators
        est = DecisionTreeRegressor(
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

        # we need to set est.tree_ field with Trees constructed from DAAL solution
        estimators_ = []
        for i in range(self.n_estimators):
            est_i = clone(est)
            est_i.n_features_ = self.n_features_
            est_i.n_outputs_ = self.n_outputs_

            tree_i_state_class = daal4py.getTreeState(model, i)
            tree_i_state_dict = {
                'max_depth' : tree_i_state_class.max_depth,
                'node_count' : tree_i_state_class.node_count,
                'nodes' : tree_i_state_class.node_ar,
                'values': tree_i_state_class.value_ar }

            est_i.tree_ = Tree(self.n_features_, np.array([1], dtype=np.intp), self.n_outputs_)
            est_i.tree_.__setstate__(tree_i_state_dict)
            estimators_.append(est_i)

        self.estimators_ = estimators_
        # compute oob_score_
        if self.oob_score:
            self._set_oob_score(X, y)

        return self


    def daal_predict(self, X):
        check_is_fitted(self, 'daal_model_')
        X = self._validate_X_predict(X)

        dfr_alg = daal4py.decision_forest_regression_prediction(fptype='float')
        dfr_predictionResult = dfr_alg.compute(X, self.daal_model_)

        pred = dfr_predictionResult.prediction

        return pred.ravel()


    def fit(self, X, y):
        return self.daal_fit(X, y)


    def predict(self, X):
        return self.daal_predict(X)
