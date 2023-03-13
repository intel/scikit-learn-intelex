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
from warnings import warn

from mpi4py import MPI
from dpctl import SyclQueue
from sklearnex.spmd.linear_model import LinearRegression


def generate_X_y(ns, data_seed):
    nf, nr = 129, 131

    crng = np.random.default_rng(777)
    coef = crng.uniform(-4, 1, size=(nr, nf)).T
    intp = crng.uniform(-1, 9, size=(nr, ))

    drng = np.random.default_rng(data_seed)
    data = drng.uniform(-7, 7, size=(ns, nf))
    resp = data @ coef + intp[np.newaxis, :]

    return data, resp


def get_train_data(rank):
    return generate_X_y(101, rank)


def get_test_data(rank):
    return generate_X_y(1024, rank)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size < 2:
    warn("This example was intentionally "
         "designed to run in distributed mode only", RuntimeWarning)

X, y = get_train_data(rank)

queue = SyclQueue("gpu")

model = LinearRegression().fit(X, y, queue)

print(f"Coefficients on rank {rank}:\n", model.coef_)
print(f"Intercept on rank {rank}:\n", model.intercept_)

X_test, _ = get_test_data(rank)

result = model.predict(X_test, queue)

print(f"Result on rank {rank}:\n", result)
