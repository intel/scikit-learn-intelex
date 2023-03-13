
#===============================================================================
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
#===============================================================================

import numpy as np
from mpi4py import MPI
import dpctl
from numpy.testing import assert_allclose
from onedal.spmd.decomposition import PCA as PCASpmd
from sklearn.decomposition import PCA as PCAstock

if __name__ == "__main__":
    q = dpctl.SyclQueue("gpu")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    params_spmd = {"ns": 15, "nf": 21, "nr": 23}

    drng = np.random.default_rng(0)
    Xsp = drng.uniform(-7, 7, size=(15, 12))
    ysp = drng.uniform(-7, 7, size=(15, 23))

    pcaspmd = PCASpmd(n_components=2).fit(Xsp, ysp, q)

    pcastock = PCAstock(n_components=2).fit(Xsp)
    assert_allclose(pcaspmd.singular_values_, pcastock.singular_values_)
