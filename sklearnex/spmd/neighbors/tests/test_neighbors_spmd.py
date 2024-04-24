# ==============================================================================
# Copyright 2024 Intel Corporation
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

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import make_regression

from onedal.tests.utils._spmd_support import (
    assert_neighbors_all_close,
    generate_classification_data,
    generate_regression_data,
    get_local_tensor,
    mpi_libs_and_gpu_available,
    spmd_assert_all_close,
)


@pytest.mark.skipif(
    not mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.mpi
def test_knncls_spmd_manual():
    # Import spmd and batch algo
    from sklearnex.neighbors import KNeighborsClassifier as KNeighborsClassifier_Batch
    from sklearnex.spmd.neighbors import KNeighborsClassifier as KNeighborsClassifier_SPMD

    # Create gold data and process into dpt
    X_train = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 2.0],
            [2.0, 0.0],
            [0.9, 1.0],
            [0.0, -1.0],
            [-1.0, 0.0],
            [-1.0, -1.0],
        ]
    )
    # TODO: handle situations where not all classes are present on all ranks?
    y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0])
    X_test = np.array(
        [
            [1.0, -0.5],
            [-5.0, 1.0],
            [0.0, 1.0],
            [10.0, -10.0],
        ]
    )

    local_dpt_X_train = get_local_tensor(X_train)
    local_dpt_y_train = get_local_tensor(y_train)
    local_dpt_X_test = get_local_tensor(X_test)

    # ensure predictions of batch algo match spmd
    spmd_model = KNeighborsClassifier_SPMD(n_neighbors=1, algorithm="brute").fit(
        local_dpt_X_train, local_dpt_y_train
    )
    batch_model = KNeighborsClassifier_Batch(n_neighbors=1, algorithm="brute").fit(
        X_train, y_train
    )
    spmd_dists, spmd_indcs = spmd_model.kneighbors(local_dpt_X_test)
    batch_dists, batch_indcs = batch_model.kneighbors(X_test)
    spmd_result = spmd_model.predict(local_dpt_X_test)
    batch_result = batch_model.predict(X_test)

    assert_neighbors_all_close(spmd_indcs, batch_indcs)
    assert_neighbors_all_close(spmd_dists, batch_dists)
    spmd_assert_all_close(spmd_result, batch_result)


@pytest.mark.skipif(
    not mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.parametrize("n_samples", [200, 10000])
@pytest.mark.parametrize("n_features_and_classes", [(5, 2), (25, 2), (25, 10)])
@pytest.mark.parametrize("n_neighbors", [1, 5, 20])
@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.mpi
def test_knncls_spmd_synthetic(
    n_samples, n_features_and_classes, n_neighbors, weights, metric="euclidean"
):
    n_features, n_classes = n_features_and_classes
    # Import spmd and batch algo
    from sklearnex.neighbors import KNeighborsClassifier as KNeighborsClassifier_Batch
    from sklearnex.spmd.neighbors import KNeighborsClassifier as KNeighborsClassifier_SPMD

    # Generate data and process into dpt
    X_train, X_test, y_train, _ = generate_classification_data(
        n_samples, n_features, n_classes
    )

    local_dpt_X_train = get_local_tensor(X_train)
    local_dpt_y_train = get_local_tensor(y_train)
    local_dpt_X_test = get_local_tensor(X_test)

    # ensure predictions of batch algo match spmd
    spmd_model = KNeighborsClassifier_SPMD(
        n_neighbors=n_neighbors, weights=weights, metric=metric, algorithm="brute"
    ).fit(local_dpt_X_train, local_dpt_y_train)
    batch_model = KNeighborsClassifier_Batch(
        n_neighbors=n_neighbors, weights=weights, metric=metric, algorithm="brute"
    ).fit(X_train, y_train)
    spmd_dists, spmd_indcs = spmd_model.kneighbors(local_dpt_X_test)
    batch_dists, batch_indcs = batch_model.kneighbors(X_test)
    spmd_result = spmd_model.predict(local_dpt_X_test)
    batch_result = batch_model.predict(X_test)

    assert_neighbors_all_close(spmd_indcs, batch_indcs)
    assert_neighbors_all_close(spmd_dists, batch_dists)
    spmd_assert_all_close(spmd_result, batch_result)


@pytest.mark.skipif(
    not mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.mpi
def test_knnreg_spmd_manual():
    # Import spmd and batch algo
    from sklearnex.neighbors import KNeighborsRegressor as KNeighborsRegressor_Batch
    from sklearnex.spmd.neighbors import KNeighborsRegressor as KNeighborsRegressor_SPMD

    # Create gold data and process into dpt
    X_train = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 2.0],
            [2.0, 0.0],
            [1.0, 1.0],
            [0.0, -1.0],
            [-1.0, 0.0],
            [-1.0, -1.0],
        ]
    )
    y_train = np.array([3.0, 5.0, 4.0, 7.0, 5.0, 6.0, 1.0, 2.0, 0.0])
    X_test = np.array(
        [
            [1.0, -0.5],
            [-5.0, 1.0],
            [0.0, 1.0],
            [10.0, -10.0],
        ]
    )

    local_dpt_X_train = get_local_tensor(X_train)
    local_dpt_y_train = get_local_tensor(y_train)
    local_dpt_X_test = get_local_tensor(X_test)

    # ensure predictions of batch algo match spmd
    spmd_model = KNeighborsRegressor_SPMD(n_neighbors=1, algorithm="brute").fit(
        local_dpt_X_train, local_dpt_y_train
    )
    batch_model = KNeighborsRegressor_Batch(n_neighbors=1, algorithm="brute").fit(
        X_train, y_train
    )
    spmd_dists, spmd_indcs = spmd_model.kneighbors(local_dpt_X_test)
    batch_dists, batch_indcs = batch_model.kneighbors(X_test)
    spmd_result = spmd_model.predict(local_dpt_X_test)
    batch_result = batch_model.predict(X_test)

    assert_neighbors_all_close(spmd_indcs, batch_indcs)
    assert_neighbors_all_close(spmd_dists, batch_dists)
    spmd_assert_all_close(spmd_result, batch_result)


@pytest.mark.skipif(
    not mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.parametrize("n_samples", [200, 10000])
@pytest.mark.parametrize("n_features", [5, 25])
@pytest.mark.parametrize("n_neighbors", [1, 5, 20])
@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize(
    "metric", ["euclidean", "manhattan", "minkowski", "chebyshev", "cosine"]
)
@pytest.mark.mpi
def test_knnreg_spmd_synthetic(n_samples, n_features, n_neighbors, weights, metric):
    # Import spmd and batch algo
    from sklearnex.neighbors import KNeighborsRegressor as KNeighborsRegressor_Batch
    from sklearnex.spmd.neighbors import KNeighborsRegressor as KNeighborsRegressor_SPMD

    # Generate data and process into dpt
    X_train, X_test, y_train, _ = generate_regression_data(n_samples, n_features)

    local_dpt_X_train = get_local_tensor(X_train)
    local_dpt_y_train = get_local_tensor(y_train)
    local_dpt_X_test = get_local_tensor(X_test)

    # ensure predictions of batch algo match spmd
    spmd_model = KNeighborsRegressor_SPMD(
        n_neighbors=n_neighbors, weights=weights, metric=metric, algorithm="brute"
    ).fit(local_dpt_X_train, local_dpt_y_train)
    batch_model = KNeighborsRegressor_Batch(
        n_neighbors=n_neighbors, weights=weights, metric=metric, algorithm="brute"
    ).fit(X_train, y_train)
    spmd_dists, spmd_indcs = spmd_model.kneighbors(local_dpt_X_test)
    batch_dists, batch_indcs = batch_model.kneighbors(X_test)
    spmd_result = spmd_model.predict(local_dpt_X_test)
    batch_result = batch_model.predict(X_test)

    assert_neighbors_all_close(spmd_indcs, batch_indcs)
    assert_neighbors_all_close(spmd_dists, batch_dists)
    spmd_assert_all_close(spmd_result, batch_result, atol=1e-4)
