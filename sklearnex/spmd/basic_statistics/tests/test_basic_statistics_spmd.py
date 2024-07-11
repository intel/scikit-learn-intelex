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
)


@pytest.mark.skipif(
    not _mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues(dataframe_filter_="dpnp,dpctl", device_filter_="gpu"))
@pytest.mark.mpi
def test_basic_stats_spmd_gold(dataframe, queue):
    # Import spmd and batch algo
    from onedal.basic_statistics import BasicStatistics as BasicStatistics_Batch
    from sklearnex.spmd.basic_statistics import BasicStatistics as BasicStatistics_SPMD

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

    local_data = _convert_to_dataframe(_get_local_tensor(data), sycl_queue=queue, target_df=dataframe)

    # ensure results of batch algo match spmd
    spmd_result = BasicStatistics_SPMD().compute(local_dpt_data)
    batch_result = BasicStatistics_Batch().compute(data)

    for option in batch_result.keys():
        assert_allclose(spmd_result[option], batch_result[option])


@pytest.mark.skipif(
    not _mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.parametrize("n_samples", [100, 10000])
@pytest.mark.parametrize("n_features", [10, 100])
@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues(dataframe_filter_="dpnp,dpctl", device_filter_="gpu"))
@pytest.mark.mpi
def test_basic_stats_spmd_synthetic(n_samples, n_features, dataframe, queue):
    # Import spmd and batch algo
    from onedal.basic_statistics import BasicStatistics as BasicStatistics_Batch
    from sklearnex.spmd.basic_statistics import BasicStatistics as BasicStatistics_SPMD

    # Generate data and process into dpt
    data = _generate_statistic_data(n_samples, n_features)

    local_dpt_data = _convert_to_dataframe(_get_local_tensor(data), sycl_queue=queue, target_df=dataframe)

    # ensure results of batch algo match spmd
    spmd_result = BasicStatistics_SPMD().compute(local_dpt_data)
    batch_result = BasicStatistics_Batch().compute(data)

    for option in batch_result.keys():
        assert_allclose(spmd_result[option], batch_result[option])
