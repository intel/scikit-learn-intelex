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

from abc import ABC

from ...common._spmd_policy import _get_spmd_policy

from onedal.ensemble import RandomForestClassifier as RandomForestClassifier_Batch
from onedal.ensemble import RandomForestRegressor as RandomForestRegressor_Batch


class BaseForestSPMD(ABC):
    def _get_policy(self, queue, *data):
        return _get_spmd_policy(queue)


class RandomForestClassifier(BaseForestSPMD, RandomForestClassifier_Batch):
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
                 voting_mode='weighted',
                 error_metric_mode='none',
                 variable_importance_mode='none',
                 algorithm='hist',
                 **kwargs):
        super().__init__(
            n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
            max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split, bootstrap=bootstrap,
            oob_score=oob_score, random_state=random_state, warm_start=warm_start,
            class_weight=class_weight, ccp_alpha=ccp_alpha, max_samples=max_samples,
            max_bins=max_bins, min_bin_size=min_bin_size, infer_mode=infer_mode,
            voting_mode=voting_mode, error_metric_mode=error_metric_mode,
            variable_importance_mode=variable_importance_mode, algorithm=algorithm)


class RandomForestRegressor(BaseForestSPMD, RandomForestRegressor_Batch):
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
                 voting_mode='weighted',
                 error_metric_mode='none',
                 variable_importance_mode='none',
                 algorithm='hist'):
        super().__init__(
            n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
            max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split, bootstrap=bootstrap,
            oob_score=oob_score, random_state=random_state, warm_start=warm_start,
            class_weight=class_weight, ccp_alpha=ccp_alpha, max_samples=max_samples,
            max_bins=max_bins, min_bin_size=min_bin_size, infer_mode=infer_mode,
            voting_mode=voting_mode, error_metric_mode=error_metric_mode,
            variable_importance_mode=variable_importance_mode, algorithm=algorithm)
