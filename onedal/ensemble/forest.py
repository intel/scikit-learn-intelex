# ===============================================================================
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
# ===============================================================================

from daal4py.sklearn._utils import (
    daal_check_version, sklearn_check_version)
from sklearn.ensemble import BaseEnsemble
from abc import ABCMeta, abstractmethod
import numbers
from numbers import Number
import warnings
from sklearn.exceptions import DataConversionWarning
from sklearn.utils import (
    check_random_state,
    compute_sample_weight,
    check_array,
    deprecated)
from sklearn.utils.validation import (
    check_is_fitted,
    check_consistent_length,
    _num_samples)
from math import ceil

import numpy as np
from scipy import sparse as sp
from ..datatypes import (
    _validate_targets,
    _check_X_y,
    _check_array,
    _column_or_1d,
    _check_n_features,
    _convert_to_supported
)

from ..common._mixin import ClassifierMixin, RegressorMixin
from ..common._policy import _get_policy
from ..common._estimator_checks import _check_is_fitted
from ..datatypes._data_conversion import from_table, to_table
from onedal import _backend

from sklearn.tree import DecisionTreeClassifier


class BaseForest(BaseEnsemble, metaclass=ABCMeta):
    @abstractmethod
    def __init__(
            self,
            n_estimators,
            criterion,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            min_weight_fraction_leaf,
            max_features,
            max_leaf_nodes,
            min_impurity_decrease,
            min_impurity_split,
            bootstrap,
            oob_score,
            random_state,
            warm_start,
            class_weight,
            ccp_alpha,
            max_samples,
            max_bins,
            min_bin_size,
            infer_mode,
            splitter_mode,
            voting_mode,
            error_metric_mode,
            variable_importance_mode,
            algorithm,
            **kwargs):
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.max_samples = max_samples
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.ccp_alpha = ccp_alpha
        self.max_bins = max_bins
        self.min_bin_size = min_bin_size
        self.infer_mode = infer_mode
        self.splitter_mode = splitter_mode
        self.voting_mode = voting_mode
        self.error_metric_mode = error_metric_mode
        self.variable_importance_mode = variable_importance_mode
        self.algorithm = algorithm

    def _to_absolute_max_features(self, max_features, n_features,
                                  is_classification=False):
        if max_features is None:
            return n_features
        if isinstance(max_features, str):
            if max_features == "auto":
                if not sklearn_check_version('1.3'):
                    if sklearn_check_version('1.1'):
                        warnings.warn(
                            "`max_features='auto'` has been deprecated in 1.1 "
                            "and will be removed in 1.3. To keep the past behaviour, "
                            "explicitly set `max_features=1.0` or remove this "
                            "parameter as it is also the default value for "
                            "RandomForestRegressors and ExtraTreesRegressors.",
                            FutureWarning, )
                    return max(1, int(np.sqrt(n_features))
                               ) if is_classification else n_features
            if max_features == 'sqrt':
                return max(1, int(np.sqrt(n_features)))
            if max_features == "log2":
                return max(1, int(np.log2(n_features)))
            allowed_string_values = '"sqrt" or "log2"' if sklearn_check_version(
                '1.3') else '"auto", "sqrt" or "log2"'
            raise ValueError(
                'Invalid value for max_features. Allowed string '
                f'values are {allowed_string_values}.')
        if isinstance(max_features, (numbers.Integral, np.integer)):
            return max_features
        if max_features > 0.0:
            return max(1, int(max_features * n_features))
        return 0

    def _get_observations_per_tree_fraction(self, n_samples, max_samples):
        if max_samples is None:
            return 1.

        if isinstance(max_samples, numbers.Integral):
            if not sklearn_check_version('1.2'):
                if not (1 <= max_samples <= n_samples):
                    msg = "`max_samples` must be in range 1 to {} but got value {}"
                    raise ValueError(msg.format(n_samples, max_samples))
            else:
                if max_samples > n_samples:
                    msg = "`max_samples` must be <= n_samples={} but got value {}"
                    raise ValueError(msg.format(n_samples, max_samples))
            return float(max_samples / n_samples)

        if isinstance(max_samples, numbers.Real):
            if sklearn_check_version('1.2'):
                pass
            elif sklearn_check_version('1.0'):
                if not (0 < float(max_samples) <= 1):
                    msg = "`max_samples` must be in range (0.0, 1.0] but got value {}"
                    raise ValueError(msg.format(max_samples))
            else:
                if not (0 < float(max_samples) < 1):
                    msg = "`max_samples` must be in range (0, 1) but got value {}"
                    raise ValueError(msg.format(max_samples))
            return float(max_samples)

        msg = "`max_samples` should be int or float, but got type '{}'"
        raise TypeError(msg.format(type(max_samples)))

    def _get_onedal_params(self, data):
        n_samples, n_features = data.shape
        features_per_node = self._to_absolute_max_features(
            self.max_features, n_features, self.is_classification)

        observations_per_tree_fraction = self._get_observations_per_tree_fraction(
            n_samples=n_samples, max_samples=self.max_samples)
        observations_per_tree_fraction = observations_per_tree_fraction if bool(
            self.bootstrap) else 1.

        if not self.bootstrap and self.max_samples is not None:
            raise ValueError(
                "`max_sample` cannot be set if `bootstrap=False`. "
                "Either switch to `bootstrap=True` or set "
                "`max_sample=None`."
            )
        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available"
                             " if bootstrap=True")

        min_observations_in_leaf_node = (
            self.min_samples_leaf if isinstance(
                self.min_samples_leaf,
                numbers.Integral) else int(
                ceil(
                    self.min_samples_leaf * n_samples)))

        min_observations_in_split_node = (
            self.min_samples_split if isinstance(
                self.min_samples_split,
                numbers.Integral) else int(
                ceil(
                    self.min_samples_split * n_samples)))

        onedal_params = {
            'fptype': 'float' if data.dtype == np.float32 else 'double',
            'method': self.algorithm,
            'infer_mode': self.infer_mode,
            'voting_mode': self.voting_mode,
            'observations_per_tree_fraction': observations_per_tree_fraction,
            'impurity_threshold': float(
                0.0 if self.min_impurity_split is None else self.min_impurity_split),
            'min_weight_fraction_in_leaf_node': self.min_weight_fraction_leaf,
            'min_impurity_decrease_in_split_node': self.min_impurity_decrease,
            'tree_count': int(self.n_estimators),
            'features_per_node': features_per_node,
            'max_tree_depth': int(0 if self.max_depth is None else self.max_depth),
            'min_observations_in_leaf_node': min_observations_in_leaf_node,
            'min_observations_in_split_node': min_observations_in_split_node,
            'max_leaf_nodes': (0 if self.max_leaf_nodes is None else self.max_leaf_nodes),
            'max_bins': self.max_bins,
            'min_bin_size': self.min_bin_size,
            'memory_saving_mode': False,
            'bootstrap': bool(self.bootstrap),
            'error_metric_mode': self.error_metric_mode,
            'variable_importance_mode': self.variable_importance_mode,
        }
        if self.is_classification:
            onedal_params['class_count'] = 0 if self.classes_ is None else len(
                self.classes_)
        if daal_check_version((2023, 'P', 101)):
            onedal_params['splitter_mode'] = self.splitter_mode
        return onedal_params

    def _check_parameters(self):
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

    def _validate_targets(self, y, dtype):
        self.class_weight_ = None
        self.classes_ = None
        return _column_or_1d(y, warn=True).astype(dtype, copy=False)

    def _get_sample_weight(self, X, y, sample_weight):
        n_samples = X.shape[0]
        dtype = X.dtype
        if n_samples == 1:
            raise ValueError("n_samples=1")

        sample_weight = np.asarray([]
                                   if sample_weight is None
                                   else sample_weight, dtype=dtype)
        sample_weight = sample_weight.ravel()

        sample_weight_count = sample_weight.shape[0]
        if sample_weight_count != 0 and sample_weight_count != n_samples:
            raise ValueError("sample_weight and X have incompatible shapes: "
                             "%r vs %r\n"
                             "Note: Sparse matrices cannot be indexed w/"
                             "boolean masks (use `indices=True` in CV)."
                             % (len(sample_weight), X.shape))

        if sample_weight_count == 0:
            sample_weight = np.ones(n_samples, dtype=dtype)
        elif isinstance(sample_weight, Number):
            sample_weight = np.full(n_samples, sample_weight, dtype=dtype)
        else:
            sample_weight = _check_array(
                sample_weight, accept_sparse=False, ensure_2d=False,
                dtype=dtype, order="C"
            )
            if sample_weight.ndim != 1:
                raise ValueError("Sample weights must be 1D array or scalar")

            if sample_weight.shape != (n_samples,):
                raise ValueError("sample_weight.shape == {}, expected {}!"
                                 .format(sample_weight.shape, (n_samples,)))
        return sample_weight

    def _get_policy(self, queue, *data):
        return _get_policy(queue, *data)

    def _fit(self, X, y, sample_weight, module, queue):
        X, y = _check_X_y(
            X, y, dtype=[np.float64, np.float32],
            force_all_finite=True, accept_sparse='csr')
        y = self._validate_targets(y, X.dtype)
        sample_weight = self._get_sample_weight(X, y, sample_weight)

        self.n_features_in_ = X.shape[1]
        if not sklearn_check_version('1.0'):
            self.n_features_ = self.n_features_in_
        policy = self._get_policy(queue, X, y, sample_weight)

        X, y, sample_weight = _convert_to_supported(policy, X, y, sample_weight)
        params = self._get_onedal_params(X)
        train_result = module.train(
            policy, params, *to_table(X, y, sample_weight))
        self._onedal_model = train_result.model

        if self.oob_score:
            if self.is_classification:
                self.oob_score_ = from_table(train_result.oob_err_accuracy)[0, 0]
                self.oob_prediction_ = from_table(
                    train_result.oob_err_decision_function)
            else:
                self.oob_score_ = from_table(train_result.oob_err_r2)[0, 0]
                self.oob_prediction_ = from_table(
                    train_result.oob_err_prediction).reshape(-1)
            if np.any(self.oob_prediction_ == 0):
                warnings.warn(
                    "Some inputs do not have OOB scores. This probably means "
                    "too few trees were used to compute any reliable OOB "
                    "estimates.",
                    UserWarning,
                )

        return self

    def _create_model(self, module):
        # TODO:
        # upate error msg.
        raise NotImplementedError('Creating model is not supported.')

    def _predict(self, X, module, queue):
        _check_is_fitted(self)
        X = _check_array(X, dtype=[np.float64, np.float32],
                         force_all_finite=True, accept_sparse=False)
        _check_n_features(self, X, False)
        policy = self._get_policy(queue, X)

        model = self._onedal_model
        X = _convert_to_supported(policy, X)
        params = self._get_onedal_params(X)
        result = module.infer(policy, params, model, to_table(X))
        y = from_table(result.responses)
        return y

    def _predict_proba(self, X, module, queue):
        _check_is_fitted(self)
        X = _check_array(X, dtype=[np.float64, np.float32],
                         force_all_finite=True, accept_sparse=False)
        _check_n_features(self, X, False)
        policy = self._get_policy(queue, X)
        X = _convert_to_supported(policy, X)
        params = self._get_onedal_params(X)
        params['infer_mode'] = 'class_probabilities'

        model = self._onedal_model
        result = module.infer(policy, params, model, to_table(X))
        y = from_table(result.probabilities)
        return y


class RandomForestClassifier(ClassifierMixin, BaseForest, metaclass=ABCMeta):
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
                 random_state=None,
                 warm_start=False,
                 class_weight=None,
                 ccp_alpha=0.0,
                 max_samples=None,
                 max_bins=256,
                 min_bin_size=1,
                 infer_mode='class_responses',
                 splitter_mode='best',
                 voting_mode='weighted',
                 error_metric_mode='none',
                 variable_importance_mode='none',
                 algorithm='hist',
                 **kwargs):
        super().__init__(
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
            random_state=random_state,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
            max_bins=max_bins,
            min_bin_size=min_bin_size,
            infer_mode=infer_mode,
            splitter_mode=splitter_mode,
            voting_mode=voting_mode,
            error_metric_mode=error_metric_mode,
            variable_importance_mode=variable_importance_mode,
            algorithm=algorithm)
        self.is_classification = True

    def _validate_targets(self, y, dtype):
        y, self.class_weight_, self.classes_ = _validate_targets(
            y, self.class_weight, dtype)

        # Decapsulate classes_ attributes
        # TODO:
        # align with `n_classes_` and `classes_` attr with daal4py implementations.
        # if hasattr(self, "classes_"):
        #    self.n_classes_ = self.classes_
        return y

    def fit(self, X, y, sample_weight=None, queue=None):
        return self._fit(X, y, sample_weight,
                         _backend.decision_forest.classification, queue)

    def predict(self, X, queue=None):
        pred = super()._predict(X, _backend.decision_forest.classification, queue)

        return np.take(
            self.classes_,
            pred.ravel().astype(
                np.int64,
                casting='unsafe'))

    def predict_proba(self, X, queue=None):
        return super()._predict_proba(X, _backend.decision_forest.classification, queue)


class RandomForestRegressor(RegressorMixin, BaseForest, metaclass=ABCMeta):
    def __init__(self,
                 n_estimators=100,
                 criterion="squared_error",
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
                 random_state=None,
                 warm_start=False,
                 class_weight=None,
                 ccp_alpha=0.0,
                 max_samples=None,
                 max_bins=256,
                 min_bin_size=1,
                 infer_mode='class_responses',
                 splitter_mode='best',
                 voting_mode='weighted',
                 error_metric_mode='none',
                 variable_importance_mode='none',
                 algorithm='hist',
                 **kwargs):
        super().__init__(
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
            random_state=random_state,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
            max_bins=max_bins,
            min_bin_size=min_bin_size,
            infer_mode=infer_mode,
            splitter_mode=splitter_mode,
            voting_mode=voting_mode,
            error_metric_mode=error_metric_mode,
            variable_importance_mode=variable_importance_mode,
            algorithm=algorithm)
        self.is_classification = False

    def fit(self, X, y, sample_weight=None, queue=None):
        if sample_weight is not None:
            if hasattr(sample_weight, '__array__'):
                sample_weight[sample_weight == 0.0] = 1.0
            sample_weight = [sample_weight]
        return super()._fit(X, y, sample_weight,
                            _backend.decision_forest.regression, queue)

    def predict(self, X, queue=None):
        return super()._predict(X, _backend.decision_forest.regression, queue).ravel()
