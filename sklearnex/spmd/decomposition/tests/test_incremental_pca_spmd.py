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

from onedal.tests.utils._dataframes_support import (
    _convert_to_dataframe,
    get_dataframes_and_queues,
)
from sklearnex.tests._utils_spmd import (
    _generate_statistic_data,
    _get_local_tensor,
    _mpi_libs_and_gpu_available,
    _spmd_assert_allclose,
)


@pytest.mark.skipif(
    not _mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.parametrize(
    "dataframe,queue",
    get_dataframes_and_queues(dataframe_filter_="dpnp,dpctl", device_filter_="gpu"),
)
@pytest.mark.parametrize("whiten", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.mpi
def test_incremental_pca_fit_spmd_gold(dataframe, queue, whiten, dtype):
    # Import spmd and non-SPMD algo
    from sklearnex.preview.decomposition import IncrementalPCA
    from sklearnex.spmd.decomposition import IncrementalPCA as IncrementalPCA_SPMD

    # Create gold data and process into dpt
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 8.0],
            [4.0, 16.0],
            [5.0, 32.0],
            [6.0, 64.0],
        ]
    ).astype(dtype=dtype)
    dpt_X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    local_X = _get_local_tensor(X)
    local_dpt_X = _convert_to_dataframe(local_X, sycl_queue=queue, target_df=dataframe)

    incpca_spmd = IncrementalPCA_SPMD(whiten=whiten)
    incpca = IncrementalPCA(whiten=whiten)

    incpca_spmd.fit(local_dpt_X)
    incpca.fit(dpt_X)

    assert_allclose(incpca.n_components_, incpca_spmd.n_components_)
    assert_allclose(incpca.components_, incpca_spmd.components_)
    assert_allclose(incpca.singular_values_, incpca_spmd.singular_values_)
    assert_allclose(incpca.mean_, incpca_spmd.mean_)
    assert_allclose(incpca.mean_, incpca_spmd.mean_)
    assert_allclose(incpca.var_, incpca_spmd.var_)
    assert_allclose(incpca.explained_variance_, incpca_spmd.explained_variance_)
    assert_allclose(
        incpca.explained_variance_ratio_, incpca_spmd.explained_variance_ratio_
    )


@pytest.mark.skipif(
    not _mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.parametrize(
    "dataframe,queue",
    get_dataframes_and_queues(dataframe_filter_="dpnp,dpctl", device_filter_="gpu"),
)
@pytest.mark.parametrize("whiten", [True, False])
@pytest.mark.parametrize("num_blocks", [1, 2])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.mpi
def test_incremental_pca_partial_fit_spmd_gold(
    dataframe, queue, whiten, num_blocks, dtype
):
    # Import spmd and non-SPMD algo
    from sklearnex.preview.decomposition import IncrementalPCA
    from sklearnex.spmd.decomposition import IncrementalPCA as IncrementalPCA_SPMD

    # Create gold data and process into dpt
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 8.0],
            [4.0, 16.0],
            [5.0, 32.0],
            [6.0, 64.0],
        ]
    ).astype(dtype=dtype)
    dpt_X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    local_X = _get_local_tensor(X)
    split_local_X = np.array_split(local_X, num_blocks)

    y = np.dot(X, [1, 2]) + 3
    dpt_y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    local_y = _get_local_tensor(y)
    split_local_y = np.array_split(local_y, num_blocks)

    incpca_spmd = IncrementalPCA_SPMD(whiten=whiten)
    incpca = IncrementalPCA(whiten=whiten)

    for i in range(num_blocks):
        local_dpt_X = _convert_to_dataframe(
            split_local_X[i], sycl_queue=queue, target_df=dataframe
        )
        local_dpt_y = _convert_to_dataframe(
            split_local_y[i], sycl_queue=queue, target_df=dataframe
        )
        incpca_spmd.partial_fit(local_dpt_X, local_dpt_y)

    incpca.fit(dpt_X, dpt_y)

    assert_allclose(incpca.n_components_, incpca_spmd.n_components_)
    assert_allclose(incpca.components_, incpca_spmd.components_)
    assert_allclose(incpca.singular_values_, incpca_spmd.singular_values_)
    assert_allclose(incpca.mean_, incpca_spmd.mean_)
    assert_allclose(incpca.mean_, incpca_spmd.mean_)
    assert_allclose(incpca.var_, incpca_spmd.var_)
    assert_allclose(incpca.explained_variance_, incpca_spmd.explained_variance_)
    assert_allclose(
        incpca.explained_variance_ratio_, incpca_spmd.explained_variance_ratio_
    )


@pytest.mark.skipif(
    not _mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.parametrize(
    "dataframe,queue",
    get_dataframes_and_queues(dataframe_filter_="dpnp,dpctl", device_filter_="gpu"),
)
@pytest.mark.parametrize("whiten", [True, False])
@pytest.mark.parametrize("n_components", [None, 2, 5])
@pytest.mark.parametrize("num_samples", [100, 200])
@pytest.mark.parametrize("num_features", [10, 20])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.mpi
def test_incremental_pca_fit_spmd_random(
    dataframe, queue, whiten, n_components, num_samples, num_features, dtype
):
    # Import spmd and non-SPMD algo
    from sklearnex.preview.decomposition import IncrementalPCA
    from sklearnex.spmd.decomposition import IncrementalPCA as IncrementalPCA_SPMD

    # Create data and process into dpt
    X = _generate_statistic_data(num_samples, num_features, dtype)
    dpt_X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    X_test = _generate_statistic_data(num_samples // 5, num_features, dtype)
    dpt_X_test = _convert_to_dataframe(X_test, sycl_queue=queue, target_df=dataframe)
    local_X = _get_local_tensor(X)
    local_dpt_X = _convert_to_dataframe(local_X, sycl_queue=queue, target_df=dataframe)

    incpca_spmd = IncrementalPCA_SPMD(n_components=n_components, whiten=whiten)
    incpca = IncrementalPCA(n_components=n_components, whiten=whiten)

    incpca_spmd.fit(local_dpt_X)
    incpca.fit(dpt_X)

    assert_allclose(incpca.n_components_, incpca_spmd.n_components_)
    assert_allclose(incpca.components_, incpca_spmd.components_)
    assert_allclose(incpca.singular_values_, incpca_spmd.singular_values_)
    assert_allclose(incpca.mean_, incpca_spmd.mean_)
    assert_allclose(incpca.mean_, incpca_spmd.mean_)
    assert_allclose(incpca.var_, incpca_spmd.var_)
    assert_allclose(incpca.explained_variance_, incpca_spmd.explained_variance_)
    assert_allclose(
        incpca.explained_variance_ratio_, incpca_spmd.explained_variance_ratio_
    )

    y_trans_spmd = incpca_spmd.transform(dpt_X_test)
    y_trans = incpca.transform(dpt_X_test)

    _spmd_assert_allclose(y_trans_spmd, y_trans)


@pytest.mark.skipif(
    not _mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.parametrize(
    "dataframe,queue",
    get_dataframes_and_queues(dataframe_filter_="dpnp,dpctl", device_filter_="gpu"),
)
@pytest.mark.parametrize("whiten", [True, False])
@pytest.mark.parametrize("n_components", [None, 2, 5])
@pytest.mark.parametrize("num_blocks", [1, 2])
@pytest.mark.parametrize("num_samples", [100, 200])
@pytest.mark.parametrize("num_features", [10, 20])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.mpi
def test_incremental_pca_partial_fit_spmd_random(
    dataframe,
    queue,
    whiten,
    n_components,
    num_blocks,
    num_samples,
    num_features,
    dtype,
):
    # Import spmd and non-SPMD algo
    from sklearnex.preview.decomposition import IncrementalPCA
    from sklearnex.spmd.decomposition import IncrementalPCA as IncrementalPCA_SPMD

    # Create data and process into dpt
    X = _generate_statistic_data(num_samples, num_features, dtype)
    dpt_X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    X_test = _generate_statistic_data(num_samples // 5, num_features, dtype)
    dpt_X_test = _convert_to_dataframe(X_test, sycl_queue=queue, target_df=dataframe)
    local_X = _get_local_tensor(X)
    X_split = np.array_split(X, num_blocks)
    split_local_X = np.array_split(local_X, num_blocks)

    incpca_spmd = IncrementalPCA_SPMD(n_components=n_components, whiten=whiten)
    incpca = IncrementalPCA(n_components=n_components, whiten=whiten)

    for i in range(num_blocks):
        local_dpt_X = _convert_to_dataframe(
            split_local_X[i], sycl_queue=queue, target_df=dataframe
        )
        dpt_X = _convert_to_dataframe(X_split[i], sycl_queue=queue, target_df=dataframe)
        incpca_spmd.partial_fit(local_dpt_X)
        incpca.partial_fit(dpt_X)

    assert_allclose(incpca.n_components_, incpca_spmd.n_components_)
    assert_allclose(incpca.components_, incpca_spmd.components_)
    assert_allclose(incpca.singular_values_, incpca_spmd.singular_values_)
    assert_allclose(incpca.mean_, incpca_spmd.mean_)
    assert_allclose(incpca.mean_, incpca_spmd.mean_)
    assert_allclose(incpca.var_, incpca_spmd.var_)
    assert_allclose(incpca.explained_variance_, incpca_spmd.explained_variance_)
    assert_allclose(
        incpca.explained_variance_ratio_, incpca_spmd.explained_variance_ratio_
    )

    y_trans_spmd = incpca_spmd.transform(dpt_X_test)
    y_trans = incpca.transform(dpt_X_test)

    _spmd_assert_allclose(y_trans_spmd, y_trans)
