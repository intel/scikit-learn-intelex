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

# sklearnex RF example for distributed systems; SPMD mode
# run like this:
#    mpirun -n 4 python ./random_forest_regressor_spmd.py

import numpy as np

import dpctl
import dpctl.tensor as dpt

from mpi4py import MPI
from sklearnex.spmd.ensemble import RandomForestRegressor

from numpy.testing import assert_allclose


def generate_X_y(par, coef_seed, data_seed):
    ns, nf = par['ns'], par['nf']

    crng = np.random.default_rng(coef_seed)
    coef = crng.uniform(-10, 10, size=(nf,))

    drng = np.random.default_rng(data_seed)
    data = drng.uniform(-100, 100, size=(ns, nf))
    resp = data @ coef

    return data, resp, coef


comm = MPI.COMM_WORLD
mpi_size = comm.Get_size()
mpi_rank = comm.Get_rank()

params_train = {'ns': 1000000, 'nf': 3}
params_test = {'ns': 100, 'nf': 3}

X_train, y_train, coef_train = generate_X_y(params_train, 10, mpi_rank)
X_test, y_test, coef_test = generate_X_y(params_test, 10, mpi_rank + 99)

assert_allclose(coef_train, coef_test)

q = dpctl.SyclQueue("gpu") # GPU

dpt_X_train = dpt.asarray(X_train, usm_type="device", sycl_queue=q)
dpt_y_train = dpt.asarray(y_train, usm_type="device", sycl_queue=q)
dpt_X_test = dpt.asarray(X_test, usm_type="device", sycl_queue=q)
# dpt_y_test = dpt.asarray(y_test, usm_type="device", sycl_queue=q)

rf = RandomForestRegressor(max_depth=2, random_state=0).fit(dpt_X_train, dpt_y_train)

# result = rf.score(X_test, y_test)

# print(f"Result on rank {mpi_rank}:\n", result)

y_predict = rf.predict(dpt_X_test)

print("Ground truth (first 5 observations on rank {}):\n{}".format(mpi_rank, y_test[:5]))
print("Regression results (first 5 observations on rank {}):\n{}"
      .format(mpi_rank, dpt.to_numpy(y_predict)[:5]))
print("RMSE:", mpi_rank, np.sqrt(np.mean((y_test - dpt.to_numpy(y_predict)) ** 2)))
