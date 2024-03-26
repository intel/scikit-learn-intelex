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

import pytest
import numpy as np
from numpy.testing import assert_allclose

try:
    import dpctl
    import mpi4py

    mpi_libs_available = True
    gpu_is_available = dpctl.has_gpu_devices()
except (ImportError, ModuleNotFoundError):
    mpi_libs_available = False


@pytest.mark.skipif(
    not mpi_libs_available or not gpu_is_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.mpi
def test_easy():
    # Import spmd algo and necessary modules
    from sklearnex.spmd.basic_statistics import BasicStatistics as BasicStatistics_SPMD
    from onedal.basic_statistics import BasicStatistics as BasicStatistics_Batch
    from mpi4py import MPI
    import dpctl.tensor as dpt
    from dpctl import SyclQueue

    # create sycl queue and gather communicator details
    q = SyclQueue("gpu")
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    data = np.array([[0., 0., 0.], [0., 1., 2.], [0., 2., 4.], [0., 3., 8.], [0., 4., 16.], [0., 5., 32.], [0., 6., 64.],])
    data_rows, data_cols = data.shape

    # divide data across ranks and move to dpt tensor
    local_start = rank * data_rows // size
    local_end = (1 + rank) * data_rows // size
    local_data = data[local_start:local_end]

    local_dpt_data = dpt.asarray(local_data, usm_type="device", sycl_queue=q)

    # ensure results of batch algo match spmd
    spmd_result = BasicStatistics_SPMD().compute(local_dpt_data)
    batch_result = BasicStatistics_Batch().compute(data)

    for option in batch_result.keys():
        assert_allclose(spmd_result[option], batch_result[option])
