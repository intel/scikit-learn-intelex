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
from numpy.testing import assert_allclose
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

try:
    import dpctl
    import dpctl.tensor as dpt
    from mpi4py import MPI

    mpi_libs_available = True
    gpu_is_available = dpctl.has_gpu_devices()
except (ImportError, ModuleNotFoundError):
    mpi_libs_available = False

mpi_libs_and_gpu_available = mpi_libs_available and gpu_is_available


def get_local_tensor(full_data, data_parallel=True):
    from dpctl import SyclQueue

    # create sycl queue and gather communicator details
    q = SyclQueue("gpu")
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    data_rows = full_data.shape[0]

    # divide data across ranks and move to dpt tensor
    local_start = rank * data_rows // size
    local_end = (1 + rank) * data_rows // size
    local_data = full_data[local_start:local_end]

    if not data_parallel:
        return local_data

    local_dpt_data = dpt.asarray(local_data, usm_type="device", sycl_queue=q)

    return local_dpt_data


def generate_regression_data(n_samples, n_features):
    X, y = make_regression(n_samples=n_samples, n_features=n_features, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    return X_train, X_test, y_train, y_test

def generate_statistic_data(n_samples, n_features):
    gen = np.random.default_rng(42)
    data = gen.uniform(low=-0.3, high=+0.7, size=(n_samples, n_features))
    return data

def spmd_assert_all_close(spmd_result, batch_result):
    # extract chunk from batch result to match with local spmd result
    local_batch_result = get_local_tensor(batch_result, data_parallel=False)
    # TODO: should this need to be converted to numpy? and if so why not for basic stats/cov
    numpy_spmd_result = dpt.to_numpy(spmd_result)

    assert_allclose(numpy_spmd_result, local_batch_result)
