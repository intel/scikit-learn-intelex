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

from daal4py.sklearn.ensemble import RandomForestClassifier as daal4py_RandomForestClassifier
from daal4py.sklearn.ensemble import RandomForestRegressor as daal4py_RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier as sklearn_RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor as sklearn_RandomForestRegressor

from ..preview.ensemble import RandomForestClassifier as preview_RandomForestClassifier
from ..preview.ensemble import RandomForestRegressor as preview_RandomForestRegressor

from .._device_offload import dispatch, wrap_output_data


class RandomForestClassifier(sklearn_RandomForestClassifier, daal4py_RandomForestClassifier, preview_RandomForestClassifier):
    __doc__ = preview_RandomForestClassifier.__doc__

#    if sklearn_check_version("1.0"):
#
#        def __init__(
#            self,
#            n_estimators=100,
#            criterion="gini",
#            max_depth=None,
#            min_samples_split=2,
#            min_samples_leaf=1,
#            min_weight_fraction_leaf=0.0,
#            max_features="sqrt" if sklearn_check_version("1.1") else "auto",
#            max_leaf_nodes=None,
#            min_impurity_decrease=0.0,
#            bootstrap=True,
#            oob_score=False,
#            n_jobs=None,
#            random_state=None,
#            verbose=0,
#            warm_start=False,
#            class_weight=None,
#            ccp_alpha=0.0,
#            max_samples=None,
#            max_bins=256,
#            min_bin_size=1,
#            splitter_mode="best",
#        ):
#            super(preview_RandomForestClassifier, self).__init__(
#                n_estimators=n_estimators,
#                criterion=criterion,
#                max_depth=max_depth,
#                min_samples_split=min_samples_split,
#                min_samples_leaf=min_samples_leaf,
#                min_weight_fraction_leaf=min_weight_fraction_leaf,
#                max_features=max_features,
#                max_leaf_nodes=max_leaf_nodes,
#                min_impurity_decrease=min_impurity_decrease,
#                bootstrap=bootstrap,
#                oob_score=oob_score,
#                n_jobs=n_jobs,
#                random_state=random_state,
#                verbose=verbose,
#                warm_start=warm_start,
#                class_weight=class_weight,
#            )
#            self.warm_start = warm_start
#            self.ccp_alpha = ccp_alpha
#            self.max_samples = max_samples
#            self.max_bins = max_bins
#            self.min_bin_size = min_bin_size
#            self.min_impurity_split = None
#            self.splitter_mode = splitter_mode
#            # self._estimator = DecisionTreeClassifier()
#
#    else:
#
#        def __init__(
#            self,
#            n_estimators=100,
#            criterion="gini",
#            max_depth=None,
#            min_samples_split=2,
#            min_samples_leaf=1,
#            min_weight_fraction_leaf=0.0,
#            max_features="auto",
#            max_leaf_nodes=None,
#            min_impurity_decrease=0.0,
#            min_impurity_split=None,
#            bootstrap=True,
#            oob_score=False,
#            n_jobs=None,
#            random_state=None,
#            verbose=0,
#            warm_start=False,
#            class_weight=None,
#            ccp_alpha=0.0,
#            max_samples=None,
#            max_bins=256,
#            min_bin_size=1,
#            splitter_mode="best",
#        ):
#            super(preview_RandomForestClassifier, self).__init__(
#                n_estimators=n_estimators,
#                criterion=criterion,
#                max_depth=max_depth,
#                min_samples_split=min_samples_split,
#                min_samples_leaf=min_samples_leaf,
#                min_weight_fraction_leaf=min_weight_fraction_leaf,
#                max_features=max_features,
#                max_leaf_nodes=max_leaf_nodes,
#                min_impurity_decrease=min_impurity_decrease,
#                min_impurity_split=min_impurity_split,
#                bootstrap=bootstrap,
#                oob_score=oob_score,
#                n_jobs=n_jobs,
#                random_state=random_state,
#                verbose=verbose,
#                warm_start=warm_start,
#                class_weight=class_weight,
#                ccp_alpha=ccp_alpha,
#                max_samples=max_samples,
#            )
#            self.warm_start = warm_start
#            self.ccp_alpha = ccp_alpha
#            self.max_samples = max_samples
#            self.max_bins = max_bins
#            self.min_bin_size = min_bin_size
#            self.min_impurity_split = None
#            self.splitter_mode = splitter_mode
#            # self._estimator = DecisionTreeClassifier()

    def fit(self, X, y, sample_weight=None):

        # dispatch(
        #     self,
        #     "fit",
        #     {
        #         "daal4py": daal4py_RandomForestClassifier.fit,
        #         "onedal": self.__class__._onedal_fit,
        #         "sklearn": sklearn_RandomForestClassifier.fit,
        #     },
        #     X,
        #     y,
        #     sample_weight,
        # )
        print(super(daal4py_RandomForestClassifier, self).fit)
        print(daal4py_RandomForestClassifier.fit)
        dispatch(
            self,
            "fit",
            {
                "daal4py": super(daal4py_RandomForestClassifier, self).fit,
                "onedal": self.__class__._onedal_fit,
                "sklearn": super(sklearn_RandomForestClassifier, self).fit,
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
                "daal4py": daal4py_RandomForestClassifier.predict,
                "onedal": self.__class__._onedal_predict,
                "sklearn": sklearn_RandomForestClassifier.predict,
            },
            X,
        )

    @wrap_output_data
    def predict_proba(self, X):

        return dispatch(
            self,
            "predict_proba",
            {
                "daal4py": daal4py_RandomForestClassifier.predict_proba,
                "onedal": self.__class__._onedal_predict_proba,
                "sklearn": sklearn_RandomForestClassifier.predict_proba,
            },
            X,
        )


class RandomForestRegressor(preview_RandomForestRegressor, daal4py_RandomForestRegressor):
    __doc__ = preview_RandomForestRegressor.__doc__

    def fit(self, X, y, sample_weight=None):

        dispatch(
            self,
            "fit",
            {
                "daal4py": daal4py_RandomForestRegressor.fit,
                "onedal": self.__class__._onedal_fit,
                "sklearn": sklearn_RandomForestRegressor.fit,
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
                "daal4py": daal4py_RandomForestRegressor.predict,
                "onedal": self.__class__._onedal_predict,
                "sklearn": sklearn_RandomForestRegressor.predict,
            },
            X,
        )
