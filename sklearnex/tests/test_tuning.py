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

import pytest
from sklearn.datasets import make_classification

from sklearnex.svm import SVC

ESTIMATORS = [SVC]
RANDOM_STATE = 42


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_tunable_estimators(estimator):
    x, y = make_classification(n_samples=400, n_features=10, random_state=RANDOM_STATE)
    estimator_instance = estimator()
    search = estimator_instance.tune(x, y, n_trials=5, random_state=RANDOM_STATE)
    assert search.best_score_ > 0.0
    assert len(search.best_params_) > 0
