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


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size < 2:
    warn("This example was intentionally "
         "designed to run in distributed mode only", RuntimeWarning)

q = dpctl.SyclQueue("gpu")

params_train = {'ns': 100000, 'nf': 8}
params_test = {'ns': 100, 'nf': 8}

X_train, y_train = generate_X_y(params_train, rank)
X_test, y_test = generate_X_y(params_test, rank + 99)

model_spmd = KNeighborsClassifier(algorithm='brute',
                                    n_neighbors=20,
                                    weights='uniform',
                                    p=2,
                                    metric='minkowski')
model_spmd.fit(X_train, y_train, queue=q)

y_predict = model_spmd.predict(X_test, queue=q)

print("Brute Force Distributed kNN classification results:")
print("Ground truth (first 5 observations on rank {}):\n{}".format(rank, y_test[:5]))
print("Classification results (first 5 observations on rank {}):\n{}"
      .format(rank, y_predict[:5]))
print("Accuracy for entire rank {} (256 classes): {}"
      .format(rank, np.mean(np.equal(y_test, y_predict))))
