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

# sklearnex RF example for distributed memory systems; SPMD mode
# run like this:
#    mpirun -n 4 python ./spmd_random_forest_regressor.py

import numpy as np

import dpctl
import dpctl.tensor as dpt

from mpi4py import MPI
from sklearnex.spmd.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


comm = MPI.COMM_WORLD
mpi_size = comm.Get_size()
mpi_rank = comm.Get_rank()

X, y = make_regression(n_samples=100, n_features=4, n_informative=2,
                       random_state=mpi_rank, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=0.2, random_state=mpi_rank)

q = dpctl.SyclQueue("gpu") # GPU

# TODO:
# sklearnex level
# dpt_X = dpt.asarray(X, usm_type="device", sycl_queue=q)
# dpt_y = dpt.asarray(y, usm_type="device", sycl_queue=q)

rf = RandomForestRegressor(max_depth=2, random_state=0).fit(X_train,y_train, queue=q)

result = rf.score(X_test, y_test, queue=q)

print(f"Result on rank {mpi_rank}:\n", result)

y_predict = rf.predict(X_test, queue=q)

print(np.mean(np.equal(y_test, y_predict)))
print(y_test[:5])
print(y_predict[:5])
