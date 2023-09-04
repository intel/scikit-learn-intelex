#!/usr/bin/env python
# ==============================================================================
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
# ==============================================================================

import numbers
import warnings
from abc import ABC

import numpy as np
from scipy import sparse as sp
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier as sklearn_ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor as sklearn_ExtraTreesRegressor
from sklearn.exceptions import DataConversionWarning
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import ExtraTreeClassifier, ExtraTreeRegressor
from sklearn.tree._tree import Tree
from sklearn.utils import check_random_state, deprecated
from sklearn.utils.validation import (
    check_array,
    check_consistent_length,
    check_is_fitted,
    check_X_y,
)

from daal4py.sklearn._utils import (
    check_tree_nodes,
    daal_check_version,
    sklearn_check_version,
)
from onedal.ensemble import ExtraTreesClassifier as onedal_ExtraTreesClassifier
from onedal.ensemble import ExtraTreesRegressor as onedal_ExtraTreesRegressor
from onedal.ensemble import RandomForestClassifier as onedal_RandomForestClassifier
from onedal.ensemble import RandomForestRegressor as onedal_RandomForestRegressor

# try catch needed for changes in structures observed in Scikit-learn around v0.22
try:
    from sklearn.ensemble._forest import ForestClassifier as sklearn_ForestClassifier
    from sklearn.ensemble._forest import ForestRegressor as sklearn_ForestRegressor
except ModuleNotFoundError:
    from sklearn.ensemble.forest import ForestClassifier as sklearn_ForestClassifier
    from sklearn.ensemble.forest import ForestRegressor as sklearn_ForestRegressor
from onedal.primitives import get_tree_state_cls, get_tree_state_reg
from onedal.utils import _num_features, _num_samples

from ..._config import get_config
from ..._device_offload import dispatch, wrap_output_data
from ..._utils import PatchingConditionsChain

if sklearn_check_version("1.2"):
    from sklearn.utils._param_validation import Interval


class BaseTree(ABC):
    def _fit_proba(self, X, y, sample_weight=None, queue=None):
        params = self.get_params()
        self.__class__(**params)

        # We use stock metaestimators below, so the only way
        # to pass a queue is using config_context.
        cfg = get_config()
        cfg["target_offload"] = queue

    def _save_attributes(self):
        self._onedal_model = self._onedal_estimator._onedal_model

        if self.oob_score:
            self.oob_score_ = self._onedal_estimator.oob_score_
            if hasattr(self._onedal_estimator, "oob_prediction_"):
                self.oob_prediction_ = self._onedal_estimator.oob_prediction_
            if hasattr(self._onedal_estimator, "oob_decision_function_"):
                self.oob_decision_function_ = (
                    self._onedal_estimator.oob_decision_function_
                )
        return self

    # TODO:
    # move to onedal modul.
    def _check_parameters(self):
        if isinstance(self.min_samples_leaf, numbers.Integral):
            if not 1 <= self.min_samples_leaf:
                raise ValueError(
                    "min_samples_leaf must be at least 1 "
                    "or in (0, 0.5], got %s" % self.min_samples_leaf
                )
        else:  # float
            if not 0.0 < self.min_samples_leaf <= 0.5:
                raise ValueError(
                    "min_samples_leaf must be at least 1 "
                    "or in (0, 0.5], got %s" % self.min_samples_leaf
                )
        if isinstance(self.min_samples_split, numbers.Integral):
            if not 2 <= self.min_samples_split:
                raise ValueError(
                    "min_samples_split must be an integer "
                    "greater than 1 or a float in (0.0, 1.0]; "
                    "got the integer %s" % self.min_samples_split
                )
        else:  # float
            if not 0.0 < self.min_samples_split <= 1.0:
                raise ValueError(
                    "min_samples_split must be an integer "
                    "greater than 1 or a float in (0.0, 1.0]; "
                    "got the float %s" % self.min_samples_split
                )
        if not 0 <= self.min_weight_fraction_leaf <= 0.5:
            raise ValueError("min_weight_fraction_leaf must in [0, 0.5]")
        if hasattr(self, "min_impurity_split"):
            warnings.warn(
                "The min_impurity_split parameter is deprecated. "
                "Its default value has changed from 1e-7 to 0 in "
                "version 0.23, and it will be removed in 0.25. "
                "Use the min_impurity_decrease parameter instead.",
                FutureWarning,
            )

            if getattr(self, "min_impurity_split") < 0.0:
                raise ValueError(
                    "min_impurity_split must be greater than " "or equal to 0"
                )
        if self.min_impurity_decrease < 0.0:
            raise ValueError(
                "min_impurity_decrease must be greater than " "or equal to 0"
            )
        if self.max_leaf_nodes is not None:
            if not isinstance(self.max_leaf_nodes, numbers.Integral):
                raise ValueError(
                    "max_leaf_nodes must be integral number but was "
                    "%r" % self.max_leaf_nodes
                )
            if self.max_leaf_nodes < 2:
                raise ValueError(
                    ("max_leaf_nodes {0} must be either None " "or larger than 1").format(
                        self.max_leaf_nodes
                    )
                )
        if isinstance(self.max_bins, numbers.Integral):
            if not 2 <= self.max_bins:
                raise ValueError("max_bins must be at least 2, got %s" % self.max_bins)
        else:
            raise ValueError(
                "max_bins must be integral number but was " "%r" % self.max_bins
            )
        if isinstance(self.min_bin_size, numbers.Integral):
            if not 1 <= self.min_bin_size:
                raise ValueError(
                    "min_bin_size must be at least 1, got %s" % self.min_bin_size
                )
        else:
            raise ValueError(
                "min_bin_size must be integral number but was " "%r" % self.min_bin_size
            )

    def check_sample_weight(self, sample_weight, X, dtype=None):
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
                order="C",
            )
            if sample_weight.ndim != 1:
                raise ValueError("Sample weights must be 1D array or scalar")

            if sample_weight.shape != (n_samples,):
                raise ValueError(
                    "sample_weight.shape == {}, expected {}!".format(
                        sample_weight.shape, (n_samples,)
                    )
                )
        return sample_weight

    @property
    def estimators_(self):
        if hasattr(self, "_cached_estimators_"):
            if self._cached_estimators_ is None and self._onedal_model:
                self._estimators_()
            return self._cached_estimators_
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute 'estimators_'"
            )

    @estimators_.setter
    def estimators_(self, estimators):
        # Needed to allow for proper sklearn operation in fallback mode
        self._cached_estimators_ = estimators


class ForestClassifier(sklearn_ForestClassifier, BaseTree):
    # Surprisingly, even though scikit-learn warns against using
    # their ForestClassifier directly, it actually has a more stable
    # API than the user-facing objects (over time). If they change it
    # significantly at some point then this may need to be versioned.

    def __init__(
        self,
        estimator,
        n_estimators=100,
        *,
        estimator_params=tuple(),
        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        max_samples=None,
    ):
        super().__init__(
            estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            max_samples=max_samples,
        )

        # The splitter is recognized here for proper dispatching.
        self._estimator = estimator  # TODO: Verify if this is done in older verions
        if self._estimator.__class__ == DecisionTreeClassifier:
            self._onedal_classifier = onedal_RandomForestClassifier
        elif self._estimator.__class__ == ExtraTreeClassifier:
            self._onedal_classifier = onedal_ExtraTreesClassifier
        else:
            raise TypeError(
                f"{estimator.__class__.__name__} is not a supported tree classifier"
            )

    def fit(self, X, y, sample_weight=None):
        dispatch(
            self,
            "fit",
            {
                "onedal": self.__class__._onedal_fit,
                "sklearn": sklearn_ForestClassifier.fit,
            },
            X,
            y,
            sample_weight,
        )
        return self

    def _onedal_fit_ready(self, patching_status, X, y, sample_weight):
        if sp.issparse(y):
            raise ValueError("sparse multilabel-indicator for y is not supported.")

        if sklearn_check_version("1.2"):
            self._validate_params()
        else:
            self._check_parameters()

        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available" " if bootstrap=True")

        patching_status.and_conditions(
            [
                (
                    self.oob_score
                    and daal_check_version((2021, "P", 500))
                    or not self.oob_score,
                    "OOB score is only supported starting from 2021.5 version of oneDAL.",
                ),
                (not sp.issparse(X), "X is sparse. Sparse input is not supported."),
                (
                    self.ccp_alpha == 0.0,
                    f"Non-zero 'ccp_alpha' ({self.ccp_alpha}) is not supported.",
                ),
                (
                    self.criterion == "gini",
                    f"'{self.criterion}' criterion is not supported. "
                    "Only 'gini' criterion is supported.",
                ),
                (self.warm_start is False, "Warm start is not supported."),
                (
                    self.n_estimators <= 6024,
                    "More than 6024 estimators is not supported.",
                ),
            ]
        )

        if patching_status.get_status():
            if sklearn_check_version("1.0"):
                self._check_feature_names(X, reset=True)
            X = check_array(X, dtype=[np.float32, np.float64])
            y = np.asarray(y)
            y = np.atleast_1d(y)
            if y.ndim == 2 and y.shape[1] == 1:
                warnings.warn(
                    "A column-vector y was passed when a 1d array was"
                    " expected. Please change the shape of y to "
                    "(n_samples,), for example using ravel().",
                    DataConversionWarning,
                    stacklevel=2,
                )
            check_consistent_length(X, y)

            if y.ndim == 1:
                y = np.reshape(y, (-1, 1))
            self.n_outputs_ = y.shape[1]
            patching_status.and_conditions(
                [
                    (
                        self.n_outputs_ == 1,
                        f"Number of outputs ({self.n_outputs_}) is not 1.",
                    ),
                    (
                        y.dtype in [np.float32, np.float64, np.int32, np.int64],
                        f"Datatype ({y.dtype}) for y is not supported.",
                    ),
                ]
            )
            # TODO: Fix to support integers as input

            n_samples = X.shape[0]
            if isinstance(self.max_samples, numbers.Integral):
                if not sklearn_check_version("1.2"):
                    if not (1 <= self.max_samples <= n_samples):
                        msg = "`max_samples` must be in range 1 to {} but got value {}"
                        raise ValueError(msg.format(n_samples, self.max_samples))
                else:
                    if self.max_samples > n_samples:
                        msg = "`max_samples` must be <= n_samples={} but got value {}"
                        raise ValueError(msg.format(n_samples, self.max_samples))
            elif isinstance(self.max_samples, numbers.Real):
                if sklearn_check_version("1.2"):
                    pass
                elif sklearn_check_version("1.0"):
                    if not (0 < float(self.max_samples) <= 1):
                        msg = "`max_samples` must be in range (0.0, 1.0] but got value {}"
                        raise ValueError(msg.format(self.max_samples))
                else:
                    if not (0 < float(self.max_samples) < 1):
                        msg = "`max_samples` must be in range (0, 1) but got value {}"
                        raise ValueError(msg.format(self.max_samples))
            elif self.max_samples is not None:
                msg = "`max_samples` should be int or float, but got type '{}'"
                raise TypeError(msg.format(type(self.max_samples)))

            if not self.bootstrap and self.max_samples is not None:
                raise ValueError(
                    "`max_sample` cannot be set if `bootstrap=False`. "
                    "Either switch to `bootstrap=True` or set "
                    "`max_sample=None`."
                )

        return patching_status, X, y, sample_weight

    @wrap_output_data
    def predict(self, X):
        return dispatch(
            self,
            "predict",
            {
                "onedal": self.__class__._onedal_predict,
                "sklearn": sklearn_ForestClassifier.predict,
            },
            X,
        )

    @wrap_output_data
    def predict_proba(self, X):
        # TODO:
        # _check_proba()
        # self._check_proba()
        if sklearn_check_version("1.0"):
            self._check_feature_names(X, reset=False)
        if hasattr(self, "n_features_in_"):
            try:
                num_features = _num_features(X)
            except TypeError:
                num_features = _num_samples(X)
            if num_features != self.n_features_in_:
                raise ValueError(
                    (
                        f"X has {num_features} features, "
                        f"but {self.__class__.__name__} is expecting "
                        f"{self.n_features_in_} features as input"
                    )
                )
        return dispatch(
            self,
            "predict_proba",
            {
                "onedal": self.__class__._onedal_predict_proba,
                "sklearn": sklearn_ForestClassifier.predict_proba,
            },
            X,
        )

    fit.__doc__ = sklearn_ForestClassifier.fit.__doc__
    predict.__doc__ = sklearn_ForestClassifier.predict.__doc__
    predict_proba.__doc__ = sklearn_ForestClassifier.predict_proba.__doc__

    if sklearn_check_version("1.0"):

        @deprecated(
            "Attribute `n_features_` was deprecated in version 1.0 and will be "
            "removed in 1.2. Use `n_features_in_` instead."
        )
        @property
        def n_features_(self):
            return self.n_features_in_

    def _estimators_(self):
        # _estimators_ should only be called if _onedal_model exists
        check_is_fitted(self, "_onedal_model")
        classes_ = self.classes_[0]
        n_classes_ = (
            self.n_classes_ if isinstance(self.n_classes_, int) else self.n_classes_[0]
        )
        # convert model to estimators
        params = {
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "min_weight_fraction_leaf": self.min_weight_fraction_leaf,
            "max_features": self.max_features,
            "max_leaf_nodes": self.max_leaf_nodes,
            "min_impurity_decrease": self.min_impurity_decrease,
            "random_state": None,
        }
        if not sklearn_check_version("1.0"):
            params["min_impurity_split"] = self.min_impurity_split
        est = self._estimator.__class__(**params)
        # we need to set est.tree_ field with Trees constructed from Intel(R)
        # oneAPI Data Analytics Library solution
        estimators_ = []

        random_state_checked = check_random_state(self.random_state)

        for i in range(self.n_estimators):
            est_i = clone(est)
            est_i.set_params(
                random_state=random_state_checked.randint(np.iinfo(np.int32).max)
            )
            if sklearn_check_version("1.0"):
                est_i.n_features_in_ = self.n_features_in_
            else:
                est_i.n_features_ = self.n_features_in_
            est_i.n_outputs_ = self.n_outputs_
            est_i.classes_ = classes_
            est_i.n_classes_ = n_classes_
            tree_i_state_class = get_tree_state_cls(self._onedal_model, i, n_classes_)
            tree_i_state_dict = {
                "max_depth": tree_i_state_class.max_depth,
                "node_count": tree_i_state_class.node_count,
                "nodes": check_tree_nodes(tree_i_state_class.node_ar),
                "values": tree_i_state_class.value_ar,
            }
            est_i.tree_ = Tree(
                self.n_features_in_,
                np.array([n_classes_], dtype=np.intp),
                self.n_outputs_,
            )
            est_i.tree_.__setstate__(tree_i_state_dict)
            estimators_.append(est_i)

        self._cached_estimators_ = estimators_

    def _onedal_cpu_supported(self, method_name, *data):
        class_name = self.__class__.__name__
        patching_status = PatchingConditionsChain(
            f"sklearn.ensemble.{class_name}.{method_name}"
        )

        if method_name == "fit":
            patching_status, X, y, sample_weight = self._onedal_fit_ready(
                patching_status, *data
            )

            patching_status.and_conditions(
                [
                    (
                        daal_check_version((2023, "P", 200))
                        or self._estimator.__class__ == DecisionTreeClassifier,
                        "ExtraTrees only supported starting from oneDAL version 2023.2",
                    ),
                    (
                        not sp.issparse(sample_weight),
                        "sample_weight is sparse. " "Sparse input is not supported.",
                    ),
                ]
            )

            if (
                patching_status.get_status()
                and (self.random_state is not None)
                and (not daal_check_version((2024, "P", 0)))
            ):
                warnings.warn(
                    "Setting 'random_state' value is not supported. "
                    "State set by oneDAL to default value (777).",
                    RuntimeWarning,
                )

        elif method_name in ["predict", "predict_proba"]:
            X = data[0]

            patching_status.and_conditions(
                [
                    (hasattr(self, "_onedal_model"), "oneDAL model was not trained."),
                    (not sp.issparse(X), "X is sparse. Sparse input is not supported."),
                    (self.warm_start is False, "Warm start is not supported."),
                    (
                        daal_check_version((2023, "P", 100))
                        or self._estimator.__class__ == DecisionTreeClassifier,
                        "ExtraTrees only supported starting from oneDAL version 2023.2",
                    ),
                ]
            )
            if hasattr(self, "n_outputs_"):
                patching_status.and_conditions(
                    [
                        (
                            self.n_outputs_ == 1,
                            f"Number of outputs ({self.n_outputs_}) is not 1.",
                        ),
                    ]
                )

        else:
            raise RuntimeError(
                f"Unknown method {method_name} in {self.__class__.__name__}"
            )

        return patching_status

    def _onedal_gpu_supported(self, method_name, *data):
        class_name = self.__class__.__name__
        patching_status = PatchingConditionsChain(
            f"sklearn.ensemble.{class_name}.{method_name}"
        )

        if method_name == "fit":
            patching_status, X, y, sample_weight = self._onedal_fit_ready(
                patching_status, *data
            )

            patching_status.and_conditions(
                [
                    (
                        daal_check_version((2023, "P", 100))
                        or self._estimator.__class__ == DecisionTreeClassifier,
                        "ExtraTrees only supported starting from oneDAL version 2023.1",
                    ),
                    (sample_weight is not None, "sample_weight is not supported."),
                ]
            )

            if (
                patching_status.get_status()
                and (self.random_state is not None)
                and (not daal_check_version((2024, "P", 0)))
            ):
                warnings.warn(
                    "Setting 'random_state' value is not supported. "
                    "State set by oneDAL to default value (777).",
                    RuntimeWarning,
                )

        elif method_name in ["predict", "predict_proba"]:
            X = data[0]

            patching_status.and_conditions(
                [
                    (hasattr(self, "_onedal_estimator"), "oneDAL model was not trained"),
                    (
                        not sp.issparse(X),
                        "X is sparse. Sparse input is not supported.",
                    ),
                    (self.warm_start is False, "Warm start is not supported."),
                    (
                        daal_check_version((2023, "P", 100)),
                        "ExtraTrees supported starting from oneDAL version 2023.1",
                    ),
                ]
            )
            if hasattr(self, "n_outputs_"):
                patching_status.and_conditions(
                    [
                        (
                            self.n_outputs_ == 1,
                            f"Number of outputs ({self.n_outputs_}) is not 1.",
                        ),
                    ]
                )

        else:
            raise RuntimeError(
                f"Unknown method {method_name} in {self.__class__.__name__}"
            )

        return patching_status

    def _onedal_fit(self, X, y, sample_weight=None, queue=None):
        if sklearn_check_version("1.2"):
            X, y = self._validate_data(
                X,
                y,
                multi_output=False,
                accept_sparse=False,
                dtype=[np.float64, np.float32],
            )
        else:
            X, y = check_X_y(
                X,
                y,
                accept_sparse=False,
                dtype=[np.float64, np.float32],
                multi_output=False,
            )

        if sample_weight is not None:
            sample_weight = self.check_sample_weight(sample_weight, X)

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warnings.warn(
                "A column-vector y was passed when a 1d array was"
                " expected. Please change the shape of y to "
                "(n_samples,), for example using ravel().",
                DataConversionWarning,
                stacklevel=2,
            )
        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        y, expanded_class_weight = self._validate_y_class_weight(y)

        n_classes_ = self.n_classes_[0]
        self.n_features_in_ = X.shape[1]
        if not sklearn_check_version("1.0"):
            self.n_features_ = self.n_features_in_

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight
        if sample_weight is not None:
            sample_weight = [sample_weight]

        if n_classes_ < 2:
            raise ValueError("Training data only contain information about one class.")

        if self.oob_score:
            err = "out_of_bag_error_accuracy|out_of_bag_error_decision_function"
        else:
            err = "none"

        onedal_params = {
            "n_estimators": self.n_estimators,
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "min_weight_fraction_leaf": self.min_weight_fraction_leaf,
            "max_features": self.max_features,
            "max_leaf_nodes": self.max_leaf_nodes,
            "min_impurity_decrease": self.min_impurity_decrease,
            "bootstrap": self.bootstrap,
            "oob_score": self.oob_score,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "warm_start": self.warm_start,
            "error_metric_mode": err,
            "variable_importance_mode": "mdi",
            "class_weight": self.class_weight,
            "max_bins": self.max_bins,
            "min_bin_size": self.min_bin_size,
            "max_samples": self.max_samples,
        }
        if daal_check_version((2023, "P", 101)):
            onedal_params["splitter_mode"] = "random"
        if not sklearn_check_version("1.0"):
            onedal_params["min_impurity_split"] = self.min_impurity_split
        else:
            onedal_params["min_impurity_split"] = None

        # Lazy evaluation of estimators_
        self._cached_estimators_ = None

        # Compute
        self._onedal_estimator = self._onedal_classifier(**onedal_params)
        self._onedal_estimator.fit(X, np.squeeze(y), sample_weight, queue=queue)

        self._save_attributes()
        if sklearn_check_version("1.2"):
            self._estimator = ExtraTreeClassifier()

        # Decapsulate classes_ attributes
        self.n_classes_ = self.n_classes_[0]
        self.classes_ = self.classes_[0]
        return self

    def _onedal_predict(self, X, queue=None):
        X = check_array(X, dtype=[np.float32, np.float64])
        check_is_fitted(self, "_onedal_model")

        if sklearn_check_version("1.0"):
            self._check_feature_names(X, reset=False)

        res = self._onedal_estimator.predict(X, queue=queue)
        return np.take(self.classes_, res.ravel().astype(np.int64, casting="unsafe"))

    def _onedal_predict_proba(self, X, queue=None):
        X = check_array(X, dtype=[np.float64, np.float32])
        check_is_fitted(self, "_onedal_model")

        if sklearn_check_version("0.23"):
            self._check_n_features(X, reset=False)
        if sklearn_check_version("1.0"):
            self._check_feature_names(X, reset=False)
        return self._onedal_estimator.predict_proba(X, queue=queue)


class ForestRegressor(sklearn_ForestRegressor, BaseTree):
    def __init__(
        self,
        estimator,
        n_estimators=100,
        *,
        estimator_params=tuple(),
        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        max_samples=None,
    ):
        super().__init__(
            estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
        )

        # The splitter is recognized here for proper dispatching.
        self._estimator = estimator  # TODO: Verify if this is done in older verions
        if self._estimator.__class__ == DecisionTreeRegressor:
            self._onedal_regressor = onedal_RandomForestRegressor
        elif self._estimator.__class__ == ExtraTreeRegressor:
            self._onedal_regressor = onedal_ExtraTreesRegressor
        else:
            raise TypeError(
                f"{estimator.__class__.__name__} is not a supported tree regressor"
            )

    def _estimators_(self):
        # _estimators_ should only be called if _onedal_model exists
        check_is_fitted(self, "_onedal_model")
        # convert model to estimators
        params = {
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "min_weight_fraction_leaf": self.min_weight_fraction_leaf,
            "max_features": self.max_features,
            "max_leaf_nodes": self.max_leaf_nodes,
            "min_impurity_decrease": self.min_impurity_decrease,
            "random_state": None,
        }
        if not sklearn_check_version("1.0"):
            params["min_impurity_split"] = self.min_impurity_split
        est = self._estimator.__class__(**params)
        # we need to set est.tree_ field with Trees constructed from Intel(R)
        # oneAPI Data Analytics Library solution
        estimators_ = []
        random_state_checked = check_random_state(self.random_state)

        for i in range(self.n_estimators):
            est_i = clone(est)
            est_i.set_params(
                random_state=random_state_checked.randint(np.iinfo(np.int32).max)
            )
            if sklearn_check_version("1.0"):
                est_i.n_features_in_ = self.n_features_in_
            else:
                est_i.n_features_ = self.n_features_in_
            est_i.n_classes_ = 1
            est_i.n_outputs_ = self.n_outputs_
            tree_i_state_class = get_tree_state_reg(self._onedal_model, i)
            tree_i_state_dict = {
                "max_depth": tree_i_state_class.max_depth,
                "node_count": tree_i_state_class.node_count,
                "nodes": check_tree_nodes(tree_i_state_class.node_ar),
                "values": tree_i_state_class.value_ar,
            }

            est_i.tree_ = Tree(
                self.n_features_in_, np.array([1], dtype=np.intp), self.n_outputs_
            )
            est_i.tree_.__setstate__(tree_i_state_dict)
            estimators_.append(est_i)

        self._cached_estimators_ = estimators_

    def _onedal_fit_ready(self, patching_status, X, y, sample_weight):
        if sp.issparse(y):
            raise ValueError("sparse multilabel-indicator for y is not supported.")

        if sklearn_check_version("1.2"):
            self._validate_params()
        else:
            self._check_parameters()

        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available" " if bootstrap=True")

        if sklearn_check_version("1.0") and self.criterion == "mse":
            warnings.warn(
                "Criterion 'mse' was deprecated in v1.0 and will be "
                "removed in version 1.2. Use `criterion='squared_error'` "
                "which is equivalent.",
                FutureWarning,
            )

        patching_status.and_conditions(
            [
                (
                    self.oob_score
                    and daal_check_version((2021, "P", 500))
                    or not self.oob_score,
                    "OOB score is only supported starting from 2021.5 version of oneDAL.",
                ),
                (self.warm_start is False, "Warm start is not supported."),
                (
                    self.criterion in ["mse", "squared_error"],
                    f"'{self.criterion}' criterion is not supported. "
                    "Only 'mse' and 'squared_error' criteria are supported.",
                ),
                (
                    self.ccp_alpha == 0.0,
                    f"Non-zero 'ccp_alpha' ({self.ccp_alpha}) is not supported.",
                ),
                (not sp.issparse(X), "X is sparse. Sparse input is not supported."),
                (
                    self.n_estimators <= 6024,
                    "More than 6024 estimators is not supported.",
                ),
            ]
        )

        if patching_status.get_status():
            if sklearn_check_version("1.0"):
                self._check_feature_names(X, reset=True)
            X = check_array(X, dtype=[np.float64, np.float32])
            y = np.asarray(y)
            y = np.atleast_1d(y)

            if y.ndim == 2 and y.shape[1] == 1:
                warnings.warn(
                    "A column-vector y was passed when a 1d array was"
                    " expected. Please change the shape of y to "
                    "(n_samples,), for example using ravel().",
                    DataConversionWarning,
                    stacklevel=2,
                )

            y = check_array(y, ensure_2d=False, dtype=X.dtype)
            check_consistent_length(X, y)

            if y.ndim == 1:
                # reshape is necessary to preserve the data contiguity against vs
                # [:, np.newaxis] that does not.
                y = np.reshape(y, (-1, 1))

            self.n_outputs_ = y.shape[1]
            patching_status.and_conditions(
                [
                    (
                        self.n_outputs_ == 1,
                        f"Number of outputs ({self.n_outputs_}) is not 1.",
                    )
                ]
            )

            n_samples = X.shape[0]
            if isinstance(self.max_samples, numbers.Integral):
                if not sklearn_check_version("1.2"):
                    if not (1 <= self.max_samples <= n_samples):
                        msg = "`max_samples` must be in range 1 to {} but got value {}"
                        raise ValueError(msg.format(n_samples, self.max_samples))
                else:
                    if self.max_samples > n_samples:
                        msg = "`max_samples` must be <= n_samples={} but got value {}"
                        raise ValueError(msg.format(n_samples, self.max_samples))
            elif isinstance(self.max_samples, numbers.Real):
                if sklearn_check_version("1.2"):
                    pass
                elif sklearn_check_version("1.0"):
                    if not (0 < float(self.max_samples) <= 1):
                        msg = "`max_samples` must be in range (0.0, 1.0] but got value {}"
                        raise ValueError(msg.format(self.max_samples))
                else:
                    if not (0 < float(self.max_samples) < 1):
                        msg = "`max_samples` must be in range (0, 1) but got value {}"
                        raise ValueError(msg.format(self.max_samples))
            elif self.max_samples is not None:
                msg = "`max_samples` should be int or float, but got type '{}'"
                raise TypeError(msg.format(type(self.max_samples)))

            if not self.bootstrap and self.max_samples is not None:
                raise ValueError(
                    "`max_sample` cannot be set if `bootstrap=False`. "
                    "Either switch to `bootstrap=True` or set "
                    "`max_sample=None`."
                )

        return patching_status, X, y, sample_weight

    def _onedal_cpu_supported(self, method_name, *data):
        class_name = self.__class__.__name__
        patching_status = PatchingConditionsChain(
            f"sklearn.ensemble.{class_name}.{method_name}"
        )

        if method_name == "fit":
            patching_status, X, y, sample_weight = self._onedal_fit_ready(
                patching_status, *data
            )

            patching_status.and_conditions(
                [
                    (
                        daal_check_version((2023, "P", 200))
                        or self._estimator.__class__ == DecisionTreeClassifier,
                        "ExtraTrees only supported starting from oneDAL version 2023.2",
                    ),
                    (
                        not sp.issparse(sample_weight),
                        "sample_weight is sparse. " "Sparse input is not supported.",
                    ),
                ]
            )

            if (
                patching_status.get_status()
                and (self.random_state is not None)
                and (not daal_check_version((2024, "P", 0)))
            ):
                warnings.warn(
                    "Setting 'random_state' value is not supported. "
                    "State set by oneDAL to default value (777).",
                    RuntimeWarning,
                )

        elif method_name in ["predict", "predict_proba"]:
            X = data[0]

            patching_status.and_conditions(
                [
                    (hasattr(self, "_onedal_model"), "oneDAL model was not trained."),
                    (not sp.issparse(X), "X is sparse. Sparse input is not supported."),
                    (self.warm_start is False, "Warm start is not supported."),
                    (
                        daal_check_version((2023, "P", 200))
                        or self._estimator.__class__ == DecisionTreeClassifier,
                        "ExtraTrees only supported starting from oneDAL version 2023.2",
                    ),
                ]
            )
            if hasattr(self, "n_outputs_"):
                patching_status.and_conditions(
                    [
                        (
                            self.n_outputs_ == 1,
                            f"Number of outputs ({self.n_outputs_}) is not 1.",
                        ),
                    ]
                )
            else:
                dal_ready = False

        else:
            raise RuntimeError(
                f"Unknown method {method_name} in {self.__class__.__name__}"
            )

        return patching_status

    def _onedal_gpu_supported(self, method_name, *data):
        class_name = self.__class__.__name__
        patching_status = PatchingConditionsChain(
            f"sklearn.ensemble.{class_name}.{method_name}"
        )

        if method_name == "fit":
            patching_status, X, y, sample_weight = self._onedal_fit_ready(
                patching_status, *data
            )

            patching_status.and_conditions(
                [
                    (
                        daal_check_version((2023, "P", 100))
                        or self._estimator.__class__ == DecisionTreeClassifier,
                        "ExtraTrees only supported starting from oneDAL version 2023.1",
                    ),
                    (sample_weight is not None, "sample_weight is not supported."),
                ]
            )

            if (
                patching_status.get_status()
                and (self.random_state is not None)
                and (not daal_check_version((2024, "P", 0)))
            ):
                warnings.warn(
                    "Setting 'random_state' value is not supported. "
                    "State set by oneDAL to default value (777).",
                    RuntimeWarning,
                )

        elif method_name == "predict":
            X = data[0]

            patching_status.and_conditions(
                [
                    (hasattr(self, "_onedal_model"), "oneDAL model was not trained."),
                    (not sp.issparse(X), "X is sparse. Sparse input is not supported."),
                    (self.warm_start is False, "Warm start is not supported."),
                    (
                        daal_check_version((2023, "P", 100))
                        or self._estimator.__class__ == DecisionTreeClassifier,
                        "ExtraTrees only supported starting from oneDAL version 2023.1",
                    ),
                ]
            )
            if hasattr(self, "n_outputs_"):
                patching_status.and_conditions(
                    [
                        (
                            self.n_outputs_ == 1,
                            f"Number of outputs ({self.n_outputs_}) is not 1.",
                        ),
                    ]
                )

        else:
            raise RuntimeError(
                f"Unknown method {method_name} in {self.__class__.__name__}"
            )

        return patching_status

    def _onedal_fit(self, X, y, sample_weight=None, queue=None):
        if sp.issparse(y):
            raise ValueError("sparse multilabel-indicator for y is not supported.")
        if sklearn_check_version("1.2"):
            self._validate_params()
        else:
            self._check_parameters()
        if sample_weight is not None:
            sample_weight = self.check_sample_weight(sample_weight, X)
        if sklearn_check_version("1.0"):
            self._check_feature_names(X, reset=True)
        X = check_array(X, dtype=[np.float64, np.float32])
        y = np.atleast_1d(np.asarray(y))
        if y.ndim == 2 and y.shape[1] == 1:
            warnings.warn(
                "A column-vector y was passed when a 1d array was"
                " expected. Please change the shape of y to "
                "(n_samples,), for example using ravel().",
                DataConversionWarning,
                stacklevel=2,
            )
        y = check_array(y, ensure_2d=False, dtype=X.dtype)
        check_consistent_length(X, y)
        self.n_features_in_ = X.shape[1]
        if not sklearn_check_version("1.0"):
            self.n_features_ = self.n_features_in_

        if self.oob_score:
            err = "out_of_bag_error_r2|out_of_bag_error_prediction"
        else:
            err = "none"

        onedal_params = {
            "n_estimators": self.n_estimators,
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "min_weight_fraction_leaf": self.min_weight_fraction_leaf,
            "max_features": self.max_features,
            "max_leaf_nodes": self.max_leaf_nodes,
            "min_impurity_decrease": self.min_impurity_decrease,
            "bootstrap": self.bootstrap,
            "oob_score": self.oob_score,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "warm_start": self.warm_start,
            "error_metric_mode": err,
            "variable_importance_mode": "mdi",
            "max_samples": self.max_samples,
        }
        if daal_check_version((2023, "P", 101)):
            onedal_params["splitter_mode"] = "random"

        # Lazy evaluation of estimators_
        self._cached_estimators_ = None

        self._onedal_estimator = self._onedal_regressor(**onedal_params)
        self._onedal_estimator.fit(X, y, sample_weight, queue=queue)

        self._save_attributes()
        if sklearn_check_version("1.2"):
            self._estimator = ExtraTreeRegressor()

        return self

    def _onedal_predict(self, X, queue=None):
        X = check_array(X, dtype=[np.float32, np.float64])
        check_is_fitted(self, "_onedal_model")

        if sklearn_check_version("1.0"):
            self._check_feature_names(X, reset=False)

        return self._onedal_estimator.predict(X, queue=queue)

    def fit(self, X, y, sample_weight=None):
        dispatch(
            self,
            "fit",
            {
                "onedal": self.__class__._onedal_fit,
                "sklearn": sklearn_ForestRegressor.fit,
            },
            X,
            y,
            sample_weight,
        )
        return self

    @wrap_output_data
    def predict(self, X):
        return dispatch(
            self,
            "predict",
            {
                "onedal": self.__class__._onedal_predict,
                "sklearn": sklearn_ForestRegressor.predict,
            },
            X,
        )

    fit.__doc__ = sklearn_ForestRegressor.fit.__doc__
    predict.__doc__ = sklearn_ForestRegressor.predict.__doc__

    if sklearn_check_version("1.0"):

        @deprecated(
            "Attribute `n_features_` was deprecated in version 1.0 and will be "
            "removed in 1.2. Use `n_features_in_` instead."
        )
        @property
        def n_features_(self):
            return self.n_features_in_


class ExtraTreesClassifier(ForestClassifier):
    __doc__ = sklearn_ExtraTreesClassifier.__doc__

    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {
            **sklearn_ExtraTreesClassifier._parameter_constraints,
            "max_bins": [Interval(numbers.Integral, 2, None, closed="left")],
            "min_bin_size": [Interval(numbers.Integral, 1, None, closed="left")],
        }

    if sklearn_check_version("1.0"):

        def __init__(
            self,
            n_estimators=100,
            *,
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features="sqrt" if sklearn_check_version("1.1") else "auto",
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=False,
            oob_score=False,
            n_jobs=None,
            random_state=None,
            verbose=0,
            warm_start=False,
            class_weight=None,
            ccp_alpha=0.0,
            max_samples=None,
            max_bins=256,
            min_bin_size=1,
        ):
            super().__init__(
                ExtraTreeClassifier(),
                n_estimators,
                estimator_params=(
                    "criterion",
                    "max_depth",
                    "min_samples_split",
                    "min_samples_leaf",
                    "min_weight_fraction_leaf",
                    "max_features",
                    "max_leaf_nodes",
                    "min_impurity_decrease",
                    "random_state",
                    "ccp_alpha",
                ),
                bootstrap=bootstrap,
                oob_score=oob_score,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose,
                warm_start=warm_start,
                class_weight=class_weight,
                max_samples=max_samples,
            )

            self.criterion = criterion
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.min_samples_leaf = min_samples_leaf
            self.min_weight_fraction_leaf = min_weight_fraction_leaf
            self.max_features = max_features
            self.max_leaf_nodes = max_leaf_nodes
            self.min_impurity_decrease = min_impurity_decrease
            self.ccp_alpha = ccp_alpha
            self.max_bins = max_bins
            self.min_bin_size = min_bin_size

    else:

        def __init__(
            self,
            n_estimators=100,
            *,
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features="auto",
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            min_impurity_split=None,
            bootstrap=False,
            oob_score=False,
            n_jobs=None,
            random_state=None,
            verbose=0,
            warm_start=False,
            class_weight=None,
            ccp_alpha=0.0,
            max_samples=None,
            max_bins=256,
            min_bin_size=1,
        ):
            super().__init__(
                ExtraTreeClassifier(),
                n_estimators,
                estimator_params=(
                    "criterion",
                    "max_depth",
                    "min_samples_split",
                    "min_samples_leaf",
                    "min_weight_fraction_leaf",
                    "max_features",
                    "max_leaf_nodes",
                    "min_impurity_decrease",
                    "min_impurity_split",
                    "random_state",
                    "ccp_alpha",
                ),
                bootstrap=bootstrap,
                oob_score=oob_score,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose,
                warm_start=warm_start,
                class_weight=class_weight,
                max_samples=max_samples,
            )

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
            self.max_bins = max_bins
            self.min_bin_size = min_bin_size


class ExtraTreesRegressor(ForestRegressor):
    __doc__ = sklearn_ExtraTreesRegressor.__doc__

    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {
            **sklearn_ExtraTreesRegressor._parameter_constraints,
            "max_bins": [Interval(numbers.Integral, 2, None, closed="left")],
            "min_bin_size": [Interval(numbers.Integral, 1, None, closed="left")],
        }

    if sklearn_check_version("1.0"):

        def __init__(
            self,
            n_estimators=100,
            *,
            criterion="squared_error",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features=1.0 if sklearn_check_version("1.1") else "auto",
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=False,
            oob_score=False,
            n_jobs=None,
            random_state=None,
            verbose=0,
            warm_start=False,
            ccp_alpha=0.0,
            max_samples=None,
            max_bins=256,
            min_bin_size=1,
        ):
            super().__init__(
                estimator=ExtraTreeRegressor(),
                n_estimators=n_estimators,
                estimator_params=(
                    "criterion",
                    "max_depth",
                    "min_samples_split",
                    "min_samples_leaf",
                    "min_weight_fraction_leaf",
                    "max_features",
                    "max_leaf_nodes",
                    "min_impurity_decrease",
                    "random_state",
                    "ccp_alpha",
                ),
                bootstrap=bootstrap,
                oob_score=oob_score,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose,
                warm_start=warm_start,
                max_samples=max_samples,
            )

            self.criterion = criterion
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.min_samples_leaf = min_samples_leaf
            self.min_weight_fraction_leaf = min_weight_fraction_leaf
            self.max_features = max_features
            self.max_leaf_nodes = max_leaf_nodes
            self.min_impurity_decrease = min_impurity_decrease
            self.ccp_alpha = ccp_alpha
            self.max_bins = max_bins
            self.min_bin_size = min_bin_size

    else:

        def __init__(
            self,
            n_estimators=100,
            *,
            criterion="mse",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features="auto",
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            min_impurity_split=None,
            bootstrap=False,
            oob_score=False,
            n_jobs=None,
            random_state=None,
            verbose=0,
            warm_start=False,
            ccp_alpha=0.0,
            max_samples=None,
            max_bins=256,
            min_bin_size=1,
        ):
            super().__init__(
                estimator=ExtraTreeRegressor(),
                n_estimators=n_estimators,
                estimator_params=(
                    "criterion",
                    "max_depth",
                    "min_samples_split",
                    "min_samples_leaf",
                    "min_weight_fraction_leaf",
                    "max_features",
                    "max_leaf_nodes",
                    "min_impurity_decrease",
                    "min_impurity_split" "random_state",
                    "ccp_alpha",
                ),
                bootstrap=bootstrap,
                oob_score=oob_score,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose,
                warm_start=warm_start,
                max_samples=max_samples,
            )

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
