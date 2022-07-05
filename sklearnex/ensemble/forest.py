#!/usr/bin/env python
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

from abc import ABC

from .._device_offload import dispatch, wrap_output_data

from sklearn.ensemble import RandomForestClassifier as sklearn_RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor as sklearn_RandomForestRegressor
# TODO
# will be replaced with prev onedal interface
from daal4py.sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import _deprecate_positional_args

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._tree import Tree

from onedal.ensemble import RandomForestClassifier as onedal_RandomForestClassifier
from onedal.ensemble import RandomForestRegressor as onedal_RandomForestRegressor


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
        pass


class RandomForestClassifier(sklearn_RandomForestClassifier, BaseRandomForest):
    __doc__ = sklearn_RandomForestClassifier.__doc__

    @_deprecate_positional_args
    def __init__(self,
                 n_estimators=100,
                 criterion="gini", *,
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
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
                 maxBins=256,
                 minBinSize=1):
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
                bootstrap=bootstrap,
                oob_score=oob_score,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose,
                warm_start=warm_start,
                class_weight=class_weight
            )
            # TODO:
            # self.ccp_alpha = ccp_alpha
            # self.max_samples = max_samples
        self.maxBins = maxBins
        self.minBinSize = minBinSize
            # self.min_impurity_split = None


    def fit(self, X, y, sample_weight=None):
        dispatch(self, 'ensemble.RandomForestClassifier.fit', {
            'onedal': self.__class__._onedal_fit,
            'sklearn': sklearn_RandomForestClassifier.fit,
        }, X, y, sample_weight)
        return self


    def predict(self, X):
        """
        TODO:
        """
        return dispatch(self, 'ensemble.RandomForestClassifier.predict', {
            'onedal': self.__class__._onedal_predict,
            'sklearn': sklearn_RandomForestClassifier.predict,
        }, X)


    def predict_proba(self, X):
        """
        TODO:
        """
        # self._check_proba()
        return self._predict_proba


    def _predict_proba(self, X):
        # sklearn_pred_proba = (sklearn_SVC.predict_proba
        #                       if Version(sklearn_version) >= Version("1.0")
        #                       else sklearn_SVC._predict_proba)

        return dispatch(self, 'ensemble.RandomForestClassifier.predict_proba', {
            'onedal': self.__class__._onedal_predict_proba,
            'sklearn': sklearn_RandomForestClassifier._predict_proba,
        }, X)


    def _onedal_cpu_supported(self, method_name, *data):
        pass


    def _onedal_gpu_supported(self, method_name, *data):
        pass


    def _onedal_fit(self, X, y, sample_weight=None, queue=None):
        onedal_params = {
            'n_estimators': self.n_estimators,
            'criterion': self.criterion,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'min_weight_fraction_leaf':self.min_weight_fraction_leaf,
            'max_features': self.max_features,
            'max_leaf_nodes': self.max_leaf_nodes,
            'min_impurity_decrease': self.min_impurity_decrease,
            'bootstrap': self.bootstrap,
            'oob_score': self.oob_score,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'warm_start': self.warm_start,
            'class_weight': self.class_weight,
        }

        self._onedal_estimator = onedal_RandomForestClassifier(**onedal_params)
        self._onedal_estimator.fit(X, y, sample_weight, queue=queue)

        self._save_attributes()


    def _onedal_predict(self, X, queue=None):
        return self._onedal_estimator.predict(X, queue=queue)


    def _onedal_predict_proba(self, X, queue=None):
        pass


# class RandomForestRegressor(sklearn_RandomForestRegressor, BaseRandomForest):
#     __doc__ = sklearn_RandomForestRegressor.__doc__
# 
#     def __init__(self,
#                  n_estimators=100, *,
#                  criterion="squared_error",
#                  max_depth=None,
#                  min_samples_split=2,
#                  min_samples_leaf=1,
#                  min_weight_fraction_leaf=0.,
#                  max_features="auto",
#                  max_leaf_nodes=None,
#                  min_impurity_decrease=0.,
#                  bootstrap=True,
#                  oob_score=False,
#                  n_jobs=None,
#                  random_state=None,
#                  verbose=0,
#                  warm_start=False,
#                  ccp_alpha=0.0,
#                  max_samples=None,
#                  maxBins=256,
#                  minBinSize=1):
#         super().__init__(
#             n_estimators=n_estimators,
#             criterion=criterion,
#             max_depth=max_depth,
#             min_samples_split=min_samples_split,
#             min_samples_leaf=min_samples_leaf,
#             min_weight_fraction_leaf=min_weight_fraction_leaf,
#             max_features=max_features,
#             max_leaf_nodes=max_leaf_nodes,
#             min_impurity_decrease=min_impurity_decrease,
#             bootstrap=bootstrap,
#             oob_score=oob_score,
#             n_jobs=n_jobs,
#             random_state=random_state,
#             verbose=verbose,
#             warm_start=warm_start
#         )
#         self.ccp_alpha = ccp_alpha
#         self.max_samples = max_samples
#         self.maxBins = maxBins
#         self.minBinSize = minBinSize
#         self.min_impurity_split = None
# 
# 
#     def fit(self, X, y, sample_weight=None):
#         pass
# 
# 
#     def predict(self, X):
#         pass
