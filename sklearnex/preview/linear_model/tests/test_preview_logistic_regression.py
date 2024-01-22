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
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from daal4py.sklearn._utils import daal_check_version
from onedal.tests.utils._dataframes_support import (
    _as_numpy,
    _convert_to_dataframe,
    get_dataframes_and_queues,
)
from sklearnex import config_context


@pytest.mark.parametrize(
    "dataframe,queue",
    get_dataframes_and_queues(dataframe_filter_="dpnp,dpctl", device_filter_="gpu"),
)
def test_sklearnex_import(dataframe, queue):
    from sklearnex.preview.linear_model import LogisticRegression

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=42
    )
    X_train = _convert_to_dataframe(X_train, sycl_queue=queue, target_df=dataframe)
    y_train = _convert_to_dataframe(y_train, sycl_queue=queue, target_df=dataframe)
    X_test = _convert_to_dataframe(X_test, sycl_queue=queue, target_df=dataframe)

    model = LogisticRegression(fit_intercept=True, solver="newton-cg")
    model.fit(X_train, y_train)
    y_pred = _as_numpy(model.predict(X_test))
    if daal_check_version((2024, "P", 1)):
        assert "sklearnex" in model.__module__
    else:
        assert "daal4py" in model.__module__
    # in case dataframe='numpy' algorithm should fallback to sklearn
    # as cpu method is not implemented in onedal
    if dataframe != "numpy" and daal_check_version((2024, "P", 1)):
        assert hasattr(model, "_onedal_estimator")
    assert accuracy_score(y_test, y_pred) > 0.95
