# ===============================================================================
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
# ===============================================================================

import dpctl
import dpctl.tensor as dpt
import numpy as np
from mpi4py import MPI

from sklearnex.spmd.covariance import IncrementalEmpiricalCovariance


def get_local_data(data, comm):
    rank = comm.Get_rank()
    num_ranks = comm.Get_size()
    local_size = (data.shape[0] + num_ranks - 1) // num_ranks
    return data[rank * local_size : (rank + 1) * local_size]


q = dpctl.SyclQueue("gpu")
comm = MPI.COMM_WORLD
num_batches = 2

seed = 77
num_samples, num_features = 3000, 3
drng = np.random.default_rng(seed)
X = drng.random(size=(num_samples, num_features))

X_local = get_local_data(X, comm)
X_split = np.array_split(X_local, num_batches)

cov = IncrementalEmpiricalCovariance()

for i in range(num_batches):
    dpt_X = dpt.asarray(X_split[i], usm_type="device", sycl_queue=q)
    cov.partial_fit(dpt_X)

print(f"Computed covariance values on rank {comm.Get_rank()}:\n", cov.covariance_)
