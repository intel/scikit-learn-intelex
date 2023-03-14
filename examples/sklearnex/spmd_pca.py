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
from onedal.spmd.decomposition import PCA as PCASpmd


def get_data(data_seed, params_spmd):
    drng = np.random.default_rng(data_seed)
    X = drng.uniform(-7, 7, size=(params_spmd["ns"], params_spmd["nf"]))
    y = drng.uniform(-7, 7, size=(params_spmd["ns"], params_spmd["nr"]))
    return X, y


q = dpctl.SyclQueue("gpu")
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


params_spmd = {"ns": 15, "nf": 21, "nr": 23}
X_spmd, y_spmd = get_data(0, params_spmd)

pcaspmd = PCASpmd(n_components=2).fit(X_spmd, y_spmd, q)

print(f"Singular values on rank {rank}:\n", pcaspmd.singular_values_)
print(f"Explained variance Ratio on rank {rank}:\n", pcaspmd.explained_variance_ratio_)
