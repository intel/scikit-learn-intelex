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


try:
    import dpctl
    from mpi4py import MPI

    mpi_libs_available = True
    gpu_is_available = dpctl.has_gpu_devices()
except (ImportError, ModuleNotFoundError):
    mpi_libs_available = False

mpi_libs_and_gpu_available = mpi_libs_available and gpu_is_available


def get_local_dpt(full_data):
    import dpctl.tensor as dpt
    from dpctl import SyclQueue

    # create sycl queue and gather communicator details
    q = SyclQueue("gpu")
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    data_rows, data_cols = full_data.shape

    # divide data across ranks and move to dpt tensor
    local_start = rank * data_rows // size
    local_end = (1 + rank) * data_rows // size
    local_data = full_data[local_start:local_end]

    local_dpt_data = dpt.asarray(local_data, usm_type="device", sycl_queue=q)

    return local_dpt_data
