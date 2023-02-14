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

from daal4py.sklearn._utils import daal_check_version, sklearn_check_version

import dpctl

import pytest
import numpy as np

import dpctl.tensor as dpt

from mpi4py import MPI

from numpy.testing import assert_allclose

from onedal.tests.utils._device_selection import get_queues
from sklearn.linear_model import LinearRegression as LinearRegressionGtr
from onedal.spmd.linear_model import LinearRegression as LinearRegressionSpmd

from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

def split_datasets(n_ranks, *arrays):
    first = arrays[0]
    n_samples = first.shape[1]

    percentage = 1 / float(n_ranks)
    block = int(n_samples * percentage)
    assert int(block * n_ranks) <= n_samples

    n_arrays = len(arrays)
    print(n_arrays)
    results = ([],) * n_arrays

    for b in range(n_ranks):
        is_last = b + 1 == n_ranks
        for a in range(n_arrays):
            first = block * b

            if is_last:
                last = n_samples
            else:
                last = first + block

            shard = arrays[a][first:last, :]
            results[a].append(shard)

    return results

def run_spmd_training(comm, queue, params, X, y):
    rank, size = comm.Get_rank(), comm.Get_size()
    Xs, ys = split_datasets(size, X, y)

    Xc, yc = Xs[rank], ys[rank]
    print(Xc.shape, yc.shape)
    model = LinearRegressionSpmd(**params)
    model.fit(Xc, yc, queue)

    coef = model.coef_

    test = (rank + 1) % size 
    Xt, yt = Xs[test], ys[test]

    n_samples = Xt.shape[0]

    yp = model.predict(Xt, queue)
    error = mean_squared_error(yt, yp)

    return (n_samples, coef, error)

def run_groundtruth(params, X, y):
    model = LinearRegressionGtr(**params)
    model.fit(X, y)

    coef = model.coef_

    yp = model.predict(X)
    error = mean_squared_error(y, yp)

    return (coef, error)

def run_on_dataset(comm, queue, params, X, y):
    gtr_coef, gtr_err = run_groundtruth(params, X, y)
    spmd_coef, spmd_err, n_samples = run_spmd_training(comm, queue, params, X, y)

    assert_allclose(gtr_coef, spmd_coef)

    mse = np.array(spmd_err, dtype = np.float64)
    global_mse = np.zeros(comm.Get_size(), dtype = np.float64)
    comm.Allgather([mse, MPI.MPI_DOUBLE], [global_mse, MPI.MPI_DOUBLE])

    nss = np.array(n_samples, dtype = np.float64)
    global_nss = np.zeros(comm.Get_size(), dtype = np.float64)
    comm.Allgather([nss, MPI.MPI_DOUBLE], [global_nss, MPI.MPI_DOUBLE])

    mse = np.sum(global_mse * global_nss) / np.sum(global_nss)

    assert_close(mse, gtr_err)
                
def test_generated(queue):
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    n_samples, n_features = size * 10, size * 11

    print(rank, size)

    params = { 'fit_intercept' : True }
    X, y = make_regression(n_samples = n_samples, 
                               n_features = n_features)
    X, y = np.array(X), np.array(y[:, np.newaxis])
    run_on_dataset(comm, queue, params, X, y)

if __name__ == "__main__":
    q = dpctl.SyclQueue("gpu")
    test_generated(q)