#!/usr/bin/env python
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

import numpy as np
import pytest
from numpy.testing import assert_allclose

from daal4py.sklearn._utils import daal_check_version
from onedal.tests.utils._dataframes_support import (
    _as_numpy,
    _convert_to_dataframe,
    get_dataframes_and_queues,
)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import(dataframe, queue):
    from sklearnex.decomposition import PCA

    X = [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    X_copy = X.copy(deep=True)
    result_tr = [
        [-1.38340578, -0.2935787],
        [-2.22189802, 0.25133484],
        [-3.6053038, -0.04224385],
        [1.38340578, 0.2935787],
        [2.22189802, -0.25133484],
        [3.6053038, 0.04224385],
    ]

    pca = PCA(n_components=2, svd_solver="full")
    pca_fit = pca.fit(X)
    X_transformed = pca_fit.transform(X)
    X_fit_transformed = pca.fit_transform(X_copy)

    if daal_check_version((2024, "P", 100)):
        assert "sklearnex" in pca.__module__
        assert hasattr(pca_fit, "_onedal_estimator")
    else:
        assert "daal4py" in pca_fit.__module__
    assert_allclose(_as_numpy(pca_fit.singular_values_), [6.30061232, 0.54980396])
    assert_allclose(_as_numpy(X_transformed), _as_numpy(X_fit_transformed))
    assert_allclose(_as_numpy(X_transformed), result_tr)
