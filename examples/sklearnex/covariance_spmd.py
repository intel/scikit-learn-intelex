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

import dpctl
import dpctl.tensor as dpt
import numpy as np
from mpi4py import MPI

from sklearnex.spmd.covariance import EmpiricalCovariance


def get_data(data_seed):
    ns, nf = 3000, 3
    drng = np.random.default_rng(data_seed)
    X = drng.random(size=(ns, nf))
    return X


q = dpctl.SyclQueue("gpu")
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

X = get_data(rank)
dpt_X = dpt.asarray(X, usm_type="device", sycl_queue=q)

cov = EmpiricalCovariance().fit(dpt_X)

print(f"Computed covariance values on rank {rank}:\n", cov.covariance_)
