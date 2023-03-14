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
from sklearnex.spmd.neighbors import KNeighborsClassifier


def generate_X_y(par, seed):
    ns, nf = par['ns'], par['nf']

    drng = np.random.default_rng(seed)
    data = drng.uniform(-1, 1, size=(ns, nf))
    resp = (data > 0) @ (2 ** np.arange(nf))

    return data, resp


def run_example(rank,
                n_train,
                n_infer,
                n_features,
                n_neighbors=1,
                weights='uniform',
                p=2,
                metric='minkowski'):
    q = dpctl.SyclQueue("gpu")

    params_train = {'ns': n_train, 'nf': n_features}
    params_test = {'ns': n_infer, 'nf': n_features}

    X_train, y_train = generate_X_y(params_train, rank)
    X_test, y_test = generate_X_y(params_test, rank + 99)

    model_spmd = KNeighborsClassifier(algorithm='brute',
                                      n_neighbors=n_neighbors,
                                      weights=weights,
                                      p=p,
                                      metric=metric)
    model_spmd.fit(X_train, y_train, queue=q)

    y_predict = model_spmd.predict(X_test, queue=q)

    return model_spmd, y_predict, y_test


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

model_spmd, y_predict, y_test = run_example(rank, 100000, 100, 8, n_neighbors=20)

print("Brute Force Distributed kNN classification results:")
print("Ground truth (first 5 observations on rank {}):\n{}".format(rank, y_test[:5]))
print("Classification results (first 5 observations on rank {}):\n{}"
      .format(rank, y_predict[:5]))
print("Accuracy for entire rank {} (256 classes): {}"
      .format(rank, np.mean(np.equal(y_test, y_predict))))
