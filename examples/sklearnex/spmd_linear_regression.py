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
#    mpirun -n 4 python ./spmd_random_forest.py

# TODO:
# Example just for debuging. Will be reimplemented.

import numpy as np
from mpi4py import MPI
import dpctl.tensor as dpt
from onedal import _backend
from sklearn.datasets import make_regression
from sklearnex.spmd.linear_model import LinearRegression
from onedal.tests.utils._device_selection import get_queues

def run_lr_regression_with_mpi4py(q):
    comm = MPI.COMM_WORLD
    mpi_size = comm.Get_size()
    mpi_rank = comm.Get_rank()

    n_features = 1024
    X_test, y_test = make_regression(n_samples=128, 
            n_features=n_features, random_state=mpi_rank)
    X_train, y_train = make_regression(n_samples=1024, 
            n_features=n_features, random_state=mpi_rank)

    # sklearnex level
    X_dpctl = dpt.asarray(X_train, usm_type="device", sycl_queue=q)
    y_dpctl = dpt.asarray(y_train, usm_type="device", sycl_queue=q)

    # onedal interface
    model = LinearRegression(True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(mpi_rank, y_pred, y_test)

if __name__ == "__main__":
    queues = get_queues()
    for q in queues:
        run_lr_regression_with_mpi4py(q)
