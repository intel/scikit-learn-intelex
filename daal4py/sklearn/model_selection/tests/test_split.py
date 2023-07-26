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

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split as skl_train_test_split

from daal4py.sklearn._utils import daal_check_version
from daal4py.sklearn.model_selection import train_test_split as d4p_train_test_split

N_SAMPLES = [2**i + 1 for i in range(2, 17)]
RANDOM_STATE = 777


@pytest.mark.skipif(
    not daal_check_version((2021, "P", 400)),
    reason="train_test_split has bugfix since 2021.4 release",
)
@pytest.mark.parametrize("n_samples", N_SAMPLES)
def test_results_similarity(n_samples):
    x, y = make_classification(
        n_samples=n_samples, n_features=4, random_state=RANDOM_STATE
    )
    d4p_res = d4p_train_test_split(
        x,
        y,
        test_size=n_samples // 2 - 1,
        train_size=n_samples // 2 - 1,
        random_state=RANDOM_STATE,
    )
    skl_res = skl_train_test_split(
        x,
        y,
        test_size=n_samples // 2 - 1,
        train_size=n_samples // 2 - 1,
        random_state=RANDOM_STATE,
    )

    assert len(d4p_res) == len(skl_res), "train_test_splits have different output size"

    for i, _ in enumerate(d4p_res):
        assert np.all(d4p_res[i] == skl_res[i]), "train_test_splits have different output"
