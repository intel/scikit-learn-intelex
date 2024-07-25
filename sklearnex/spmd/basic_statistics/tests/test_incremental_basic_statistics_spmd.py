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

from onedal.basic_statistics.tests.test_basic_statistics import options_and_tests
from onedal.tests.utils._dataframes_support import (
    _convert_to_dataframe,
    get_dataframes_and_queues,
)
from sklearnex.tests._utils_spmd import (
    _generate_statistic_data,
    _generate_weights,
    _get_local_tensor,
    _mpi_libs_and_gpu_available,
)


@pytest.mark.skipif(
    not _mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.parametrize(
    "dataframe,queue",
    get_dataframes_and_queues(dataframe_filter_="dpnp,dpctl", device_filter_="gpu"),
)
@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.mpi
def test_incremental_basic_statistics_fit_spmd_gold(dataframe, queue, weighted, dtype):
    # Import spmd and batch algo
    from sklearnex.basic_statistics import IncrementalBasicStatistics
    from sklearnex.spmd.basic_statistics import (
        IncrementalBasicStatistics as IncrementalBasicStatistics_SPMD,
    )

    # Create gold data and process into dpt
    data = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0],
            [0.0, 2.0, 4.0],
            [0.0, 3.0, 8.0],
            [0.0, 4.0, 16.0],
            [0.0, 5.0, 32.0],
            [0.0, 6.0, 64.0],
        ]
    ).astype(dtype=dtype)

    dpt_data = _convert_to_dataframe(data, sycl_queue=queue, target_df=dataframe)

    local_dpt_data = _convert_to_dataframe(
        _get_local_tensor(data), sycl_queue=queue, target_df=dataframe
    )

    if weighted:
        weights = np.array([0, 1, 2, 3, 4, 5, 6]).astype(dtype=dtype)
        dpt_weights = _convert_to_dataframe(
            weights, sycl_queue=queue, target_df=dataframe
        )
        local_dpt_weights = _convert_to_dataframe(
            _get_local_tensor(weights), sycl_queue=queue, target_df=dataframe
        )

    # ensure results of batch algo match spmd

    if weighted:
        incbs_spmd = IncrementalBasicStatistics_SPMD().fit(
            local_dpt_data, sample_weight=local_dpt_weights
        )
        incbs = IncrementalBasicStatistics().fit(dpt_data, sample_weight=dpt_weights)

    else:
        incbs_spmd = IncrementalBasicStatistics_SPMD().fit(local_dpt_data)
        incbs = IncrementalBasicStatistics().fit(dpt_data)

    for option in options_and_tests:
        assert_allclose(getattr(incbs_spmd, option[0]), getattr(incbs, option[0]))


@pytest.mark.skipif(
    not _mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.parametrize(
    "dataframe,queue",
    get_dataframes_and_queues(dataframe_filter_="dpnp,dpctl", device_filter_="gpu"),
)
@pytest.mark.parametrize("num_blocks", [1, 2])
@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.mpi
def test_incremental_basic_statistics_partial_fit_spmd_gold(
    dataframe, queue, num_blocks, weighted, dtype
):
    # Import spmd and batch algo
    from sklearnex.basic_statistics import IncrementalBasicStatistics
    from sklearnex.spmd.basic_statistics import (
        IncrementalBasicStatistics as IncrementalBasicStatistics_SPMD,
    )

    # Create gold data and process into dpt
    data = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0],
            [0.0, 2.0, 4.0],
            [0.0, 3.0, 8.0],
            [0.0, 4.0, 16.0],
            [0.0, 5.0, 32.0],
            [0.0, 6.0, 64.0],
        ]
    )
    data = data.astype(dtype=dtype)
    dpt_data = _convert_to_dataframe(data, sycl_queue=queue, target_df=dataframe)
    local_data = _get_local_tensor(data)
    split_local_data = np.array_split(local_data, num_blocks)

    if weighted:
        weights = np.array([0, 1, 2, 3, 4, 5, 6])
        dpt_weights = _convert_to_dataframe(
            weights, sycl_queue=queue, target_df=dataframe
        )
        local_weights = _get_local_tensor(weights)
        split_local_weights = np.array_split(local_weights, num_blocks)

    incbs_spmd = IncrementalBasicStatistics_SPMD()
    incbs = IncrementalBasicStatistics()

    for i in range(num_blocks):
        local_dpt_data = _convert_to_dataframe(
            split_local_data[i], sycl_queue=queue, target_df=dataframe
        )
        if weighted:
            local_dpt_weights = _convert_to_dataframe(
                split_local_weights[i], sycl_queue=queue, target_df=dataframe
            )
            incbs_spmd.partial_fit(local_dpt_data, sample_weight=local_dpt_weights)
        else:
            incbs_spmd.partial_fit(local_dpt_data)

    if weighted:
        incbs.fit(dpt_data, sample_weight=dpt_weights)
    else:
        incbs.fit(dpt_data)

    for option in options_and_tests:
        assert_allclose(getattr(incbs_spmd, option[0]), getattr(incbs, option[0]))


@pytest.mark.skipif(
    not _mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.parametrize(
    "dataframe,queue",
    get_dataframes_and_queues(dataframe_filter_="dpnp,dpctl", device_filter_="gpu"),
)
@pytest.mark.parametrize("num_blocks", [1, 2])
@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("result_option", options_and_tests)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.mpi
def test_incremental_basic_statistics_single_option_partial_fit_spmd_gold(
    dataframe, queue, num_blocks, weighted, result_option, dtype
):
    # Import spmd and batch algo
    from sklearnex.basic_statistics import IncrementalBasicStatistics
    from sklearnex.spmd.basic_statistics import (
        IncrementalBasicStatistics as IncrementalBasicStatistics_SPMD,
    )

    # Create gold data and process into dpt
    data = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0],
            [0.0, 2.0, 4.0],
            [0.0, 3.0, 8.0],
            [0.0, 4.0, 16.0],
            [0.0, 5.0, 32.0],
            [0.0, 6.0, 64.0],
        ]
    )
    data = data.astype(dtype=dtype)
    dpt_data = _convert_to_dataframe(data, sycl_queue=queue, target_df=dataframe)
    local_data = _get_local_tensor(data)
    split_local_data = np.array_split(local_data, num_blocks)

    if weighted:
        weights = np.array([0, 1, 2, 3, 4, 5, 6])
        dpt_weights = _convert_to_dataframe(
            weights, sycl_queue=queue, target_df=dataframe
        )
        local_weights = _get_local_tensor(weights)
        split_local_weights = np.array_split(local_weights, num_blocks)

    option, _, _ = result_option
    incbs_spmd = IncrementalBasicStatistics_SPMD(result_options=option)
    incbs = IncrementalBasicStatistics(result_options=option)

    for i in range(num_blocks):
        local_dpt_data = _convert_to_dataframe(
            split_local_data[i], sycl_queue=queue, target_df=dataframe
        )
        if weighted:
            local_dpt_weights = _convert_to_dataframe(
                split_local_weights[i], sycl_queue=queue, target_df=dataframe
            )
            incbs_spmd.partial_fit(local_dpt_data, sample_weight=local_dpt_weights)
        else:
            incbs_spmd.partial_fit(local_dpt_data)

    if weighted:
        incbs.fit(dpt_data, sample_weight=dpt_weights)
    else:
        incbs.fit(dpt_data)

    assert_allclose(getattr(incbs_spmd, option), getattr(incbs, option))


@pytest.mark.skipif(
    not _mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.parametrize(
    "dataframe,queue",
    get_dataframes_and_queues(dataframe_filter_="dpnp,dpctl", device_filter_="gpu"),
)
@pytest.mark.parametrize("num_blocks", [1, 2])
@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("n_samples", [100, 10000])
@pytest.mark.parametrize("n_features", [10, 100])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.mpi
def test_incremental_basic_statistics_partial_fit_spmd_synthetic(
    dataframe, queue, num_blocks, weighted, n_samples, n_features, dtype
):
    # Import spmd and batch algo
    from sklearnex.basic_statistics import IncrementalBasicStatistics
    from sklearnex.spmd.basic_statistics import (
        IncrementalBasicStatistics as IncrementalBasicStatistics_SPMD,
    )

    # Create gold data and process into dpt
    data = _generate_statistic_data(n_samples, n_features).astype(dtype=dtype)
    local_data = _get_local_tensor(data)
    split_local_data = np.array_split(local_data, num_blocks)
    split_data = np.array_split(data, num_blocks)

    if weighted:
        weights = _generate_weights(n_samples).astype(dtype=dtype)
        local_weights = _get_local_tensor(weights)
        split_local_weights = np.array_split(local_weights, num_blocks)
        split_weights = np.array_split(weights, num_blocks)

    incbs_spmd = IncrementalBasicStatistics_SPMD()
    incbs = IncrementalBasicStatistics()

    for i in range(num_blocks):
        local_dpt_data = _convert_to_dataframe(
            split_local_data[i], sycl_queue=queue, target_df=dataframe
        )
        dpt_data = _convert_to_dataframe(
            split_data[i], sycl_queue=queue, target_df=dataframe
        )
        if weighted:
            local_dpt_weights = _convert_to_dataframe(
                split_local_weights[i], sycl_queue=queue, target_df=dataframe
            )
            dpt_weights = _convert_to_dataframe(
                split_weights[i], sycl_queue=queue, target_df=dataframe
            )
            incbs_spmd.partial_fit(local_dpt_data, sample_weight=local_dpt_weights)
            incbs.partial_fit(dpt_data, sample_weight=dpt_weights)
        else:
            incbs_spmd.partial_fit(local_dpt_data)
            incbs.partial_fit(dpt_data)

    # if weighted:
    #     incbs_spmd.fit(dpt_data, sample_weight=dpt_weights)
    # else:
    #     incbs_spmd.fit(dpt_data)

    # if weighted:
    #     incbs.fit(dpt_data, sample_weight=dpt_weights)
    # else:
    #     incbs.fit(dpt_data)

    for option in options_and_tests:
        assert_allclose(getattr(incbs_spmd, option[0]), getattr(incbs, option[0]))
