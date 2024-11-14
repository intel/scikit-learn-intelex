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
from numpy.testing import assert_allclose
from sklearn.datasets import make_classification, make_regression

from daal4py.sklearn._utils import daal_check_version
from onedal.tests.utils._dataframes_support import (
    _as_numpy,
    _convert_to_dataframe,
    get_dataframes_and_queues,
)

hparam_values = [
    (None, None, None, None),
    (8, 100, 32, 0.3),
    (16, 100, 32, 0.3),
    (32, 100, 32, 0.3),
    (64, 10, 32, 0.1),
    (128, 100, 1000, 1.0),
]


@pytest.mark.parametrize("dataframe, queue", get_dataframes_and_queues())
@pytest.mark.parametrize("block, trees, rows, scale", hparam_values)
def test_sklearnex_import_rf_classifier(dataframe, queue, block, trees, rows, scale):
    from sklearnex.ensemble import RandomForestClassifier

    X, y = make_classification(
        n_samples=1000,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=0,
        shuffle=False,
    )
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    rf = RandomForestClassifier(max_depth=2, random_state=0).fit(X, y)
    hparams = RandomForestClassifier.get_hyperparameters("infer")
    if hparams and block is not None:
        hparams.block_size = block
        hparams.min_trees_for_threading = trees
        hparams.min_number_of_rows_for_vect_seq_compute = rows
        hparams.scale_factor_for_vect_parallel_compute = scale
    assert "sklearnex" in rf.__module__
    assert_allclose([1], _as_numpy(rf.predict([[0, 0, 0, 0]])))


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import_rf_regression(dataframe, queue):
    if queue and queue.sycl_device.is_gpu:
        pytest.skip("RF regressor predict for the GPU sycl_queue is buggy.")
    from sklearnex.ensemble import RandomForestRegressor

    X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    rf = RandomForestRegressor(max_depth=2, random_state=0).fit(X, y)
    assert "sklearnex" in rf.__module__
    pred = _as_numpy(rf.predict([[0, 0, 0, 0]]))

    if queue is not None and queue.sycl_device.is_gpu:
        assert_allclose([-0.011208], pred, atol=1e-2)
    else:
        if daal_check_version((2024, "P", 0)):
            assert_allclose([-6.971], pred, atol=1e-2)
        else:
            assert_allclose([-6.839], pred, atol=1e-2)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import_et_classifier(dataframe, queue):
    if queue and queue.sycl_device.is_gpu:
        pytest.skip("ET classifier predict for the GPU sycl_queue is buggy.")
    from sklearnex.ensemble import ExtraTreesClassifier

    X, y = make_classification(
        n_samples=1000,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=0,
        shuffle=False,
    )
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    # For the 2023.2 release, random_state is not supported
    # defaults to seed=777, although it is set to 0
    rf = ExtraTreesClassifier(max_depth=2, random_state=0).fit(X, y)
    assert "sklearnex" in rf.__module__
    assert_allclose([1], _as_numpy(rf.predict([[0, 0, 0, 0]])))


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import_et_regression(dataframe, queue):
    if queue and queue.sycl_device.is_gpu:
        pytest.skip("ET regressor predict for the GPU sycl_queue is buggy.")
    from sklearnex.ensemble import ExtraTreesRegressor

    X, y = make_regression(n_features=1, random_state=0, shuffle=False)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    # For the 2023.2 release, random_state is not supported
    # defaults to seed=777, although it is set to 0
    rf = ExtraTreesRegressor(random_state=0).fit(X, y)
    assert "sklearnex" in rf.__module__
    pred = _as_numpy(
        rf.predict(
            [
                [
                    0,
                ]
            ]
        )
    )

    if queue is not None and queue.sycl_device.is_gpu:
        assert_allclose([1.909769], pred, atol=1e-2)
    else:
        assert_allclose([0.445], pred, atol=1e-2)
