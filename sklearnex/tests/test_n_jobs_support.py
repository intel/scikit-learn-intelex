# ==============================================================================
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
# ==============================================================================

import pytest

from sklearnex.cluster import KMeans
from sklearnex.linear_model import ElasticNet, Lasso, Ridge
from sklearnex.svm import SVC, SVR, NuSVC, NuSVR

estimators = [KMeans, SVC, SVR, NuSVC, NuSVR, Lasso, Ridge, ElasticNet]


@pytest.mark.parametrize("estimator", estimators)
def test_n_jobs_support(estimator):
    # use `n_jobs` parameter where original sklearn doesn't expect it
    estimator(n_jobs=1)
    # check `n_jobs` parameter doc entry
    assert "n_jobs" in estimator.__doc__
