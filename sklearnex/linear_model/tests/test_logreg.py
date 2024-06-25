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

import pytest
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from daal4py.sklearn._utils import daal_check_version
from onedal.tests.utils._dataframes_support import (
    _as_numpy,
    _convert_to_dataframe,
    get_dataframes_and_queues,
)


def prepare_input(X, y, dataframe, queue):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=42
    )
    X_train = _convert_to_dataframe(X_train, sycl_queue=queue, target_df=dataframe)
    y_train = _convert_to_dataframe(y_train, sycl_queue=queue, target_df=dataframe)
    X_test = _convert_to_dataframe(X_test, sycl_queue=queue, target_df=dataframe)
    return X_train, X_test, y_train, y_test


@pytest.mark.parametrize(
    "dataframe,queue",
    get_dataframes_and_queues(device_filter_="cpu"),
)
def test_sklearnex_multiclass_classification(dataframe, queue):
    from sklearnex.linear_model import LogisticRegression

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = prepare_input(X, y, dataframe, queue)

    logreg = LogisticRegression(fit_intercept=True, solver="lbfgs", max_iter=200).fit(
        X_train, y_train
    )

    if daal_check_version((2024, "P", 1)):
        assert "sklearnex" in logreg.__module__
    else:
        assert "daal4py" in logreg.__module__

    y_pred = _as_numpy(logreg.predict(X_test))
    assert accuracy_score(y_test, y_pred) > 0.99


@pytest.mark.parametrize(
    "dataframe,queue",
    get_dataframes_and_queues(),
)
def test_sklearnex_binary_classification(dataframe, queue):
    from sklearnex.linear_model import LogisticRegression

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = prepare_input(X, y, dataframe, queue)

    logreg = LogisticRegression(fit_intercept=True, solver="newton-cg", max_iter=100).fit(
        X_train, y_train
    )

    if daal_check_version((2024, "P", 1)):
        assert "sklearnex" in logreg.__module__
    else:
        assert "daal4py" in logreg.__module__
    if (
        dataframe != "numpy"
        and queue is not None
        and queue.sycl_device.is_gpu
        and daal_check_version((2024, "P", 1))
    ):
        # fit was done on gpu
        assert hasattr(logreg, "_onedal_estimator")

    y_pred = _as_numpy(logreg.predict(X_test))
    assert accuracy_score(y_test, y_pred) > 0.95
