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

try:
    import dpctl
    from dpctl import SyclQueue
    from mpi4py import MPI

    mpi_libs_available = True
    gpu_is_available = dpctl.has_gpu_devices()
except (ImportError, ModuleNotFoundError):
    mpi_libs_available = False

_mpi_libs_and_gpu_available = mpi_libs_available and gpu_is_available


def _get_local_tensor(full_data):
    """Splits data across ranks.

    Called on each rank to extract the subset of data assigned to that rank.

    Args:
        full_data (numpy or dpctl array): The entire set of data

    Returns:
        local_data (numpy or dpctl array): The subset of data used by the rank
    """

    # create sycl queue and gather communicator details
    q = SyclQueue("gpu")
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # divide data across ranks and move to dpt tensor
    data_rows = full_data.shape[0]
    local_start = rank * data_rows // size
    local_end = (1 + rank) * data_rows // size
    local_data = full_data[local_start:local_end]

    return local_data


def _generate_statistic_data(n_samples, n_features, random_state=42):
    # Generates statistical data
    gen = np.random.default_rng(random_state)
    data = gen.uniform(low=-0.3, high=+0.7, size=(n_samples, n_features))
    return data
