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

import dpctl
import dpctl.tensor as dpt, dpctl.memory as dpmem

from mpi4py import MPI

from onedal.tests.utils._device_selection import get_queues
from onedal import _backend

from sklearnex.spmd.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.datasets import make_classification, make_regression


def run_rf_classifier():
    X, y = make_classification(n_samples=100, n_features=4,
                               n_informative=2, n_redundant=0,
                               random_state=0, shuffle=False)
    X = X.astype("float32")
    y = y.astype("float32")

    mpi_initializer = _backend.mpi_initializer()
    mpi_initializer.init()

    q = dpctl.SyclQueue("gpu") # GPU

    # sklearnex level
    dpt_X = dpt.asarray(X, usm_type="device", sycl_queue=q)
    dpt_y = dpt.asarray(y, usm_type="device", sycl_queue=q)
    rf = RandomForestClassifier(max_depth=2, random_state=0).fit(X, y, queue=q)
    #pred = rf.predict(np.array([[0, 0, 0, 0]]), queue=q)
    pred = rf.predict(X, queue=q)
    mpi_initializer.fini()
    return pred


def run_rf_classifier_with_mpi4py():
    X, y = make_classification(n_samples=100, n_features=4,
                               n_informative=2, n_redundant=0,
                               random_state=0, shuffle=False)
    # X = X.astype("float32")
    # y = y.astype("float32")
    comm = MPI.COMM_WORLD
    mpi_size = comm.Get_size()
    mpi_rank = comm.Get_rank()

    q = dpctl.SyclQueue("gpu") # GPU

    # sklearnex level
    dpt_X = dpt.asarray(X, usm_type="device", sycl_queue=q)
    dpt_y = dpt.asarray(y, usm_type="device", sycl_queue=q)

    # sklearnex interface
    # rf = sklearex.ensemble.RandomForestClassifier(max_depth=2, random_state=0).fit(dpt_X, dpt_y)

    # onedal interface
    rf = RandomForestClassifier(max_depth=2, random_state=0).fit(X, y, queue=q)

    # sklearnex interface
    # pred = rf.predict(np.array([[0, 0, 0, 0]]))

    # onedal interface
    pred = rf.predict(np.array([[0, 0, 0, 0]]), queue=q)

    print(mpi_rank, pred)


def run_rf_regressor_with_mpi4py():
    X, y = make_regression(n_samples=100, n_features=4, n_informative=2,
                           random_state=0, shuffle=False)
    # X = X.astype("float32")
    # y = y.astype("float32")
    comm = MPI.COMM_WORLD
    mpi_size = comm.Get_size()
    mpi_rank = comm.Get_rank()

    q = dpctl.SyclQueue("gpu") # GPU

    # sklearnex level
    dpt_X = dpt.asarray(X, usm_type="device", sycl_queue=q)
    dpt_y = dpt.asarray(y, usm_type="device", sycl_queue=q)

    # sklearnex interface
    # rf = sklearex.ensemble.RandomForestRegressor(max_depth=2, random_state=0).fit(dpt_X, dpt_y)

    # onedal interface
    rf = RandomForestRegressor(max_depth=2, random_state=0).fit(X, y, queue=q)

    # sklearnex interface
    # pred = rf.predict(np.array([[0, 0, 0, 0]]))

    # onedal interface
    pred = rf.predict(np.array([[0, 0, 0, 0]]), queue=q)

    print(mpi_rank, pred)

if __name__ == "__main__":
    # run_rf_classifier()
    run_rf_classifier_with_mpi4py()
    # run_rf_regressor_with_mpi4py()