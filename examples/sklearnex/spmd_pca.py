# ===============================================================================
# Copyright 2023 Intel Corporation
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

import numpy as np
from mpi4py import MPI
import dpctl
from sklearnex.spmd.decomposition import PCA as PCASpmd


def get_data(data_seed):
    ns, nf = 15, 21
    drng = np.random.default_rng(data_seed)
    X = drng.uniform(-7, 7, size=(ns, nf))
    return X


q = dpctl.SyclQueue("gpu")
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

X_spmd = get_data(rank)

pcaspmd = PCASpmd(n_components=2).fit(X_spmd, q)

print(f"Singular values on rank {rank}:\n", pcaspmd.singular_values_)
print(f"Explained variance Ratio on rank {rank}:\n", pcaspmd.explained_variance_ratio_)
