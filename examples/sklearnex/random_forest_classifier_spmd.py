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
#    mpirun -n 4 python ./random_forest_classifier_spmd.py

import dpctl
import dpctl.tensor as dpt

import numpy as np

from mpi4py import MPI

from sklearnex.spmd.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score


def generate_X_y(par, seed):
    ns, nf = par['ns'], par['nf']

    drng = np.random.default_rng(seed)
    data = drng.uniform(-1, 1, size=(ns, nf))
    resp = (data > 0) @ (2 ** np.arange(nf))

    return data, resp


params_train = {'ns': 100000, 'nf': 8}
params_test = {'ns': 100, 'nf': 8}

comm = MPI.COMM_WORLD
mpi_size = comm.Get_size()
mpi_rank = comm.Get_rank()

X_train, y_train = generate_X_y(params_train, mpi_rank)
X_test, y_test = generate_X_y(params_test, mpi_rank + 777)

q = dpctl.SyclQueue("gpu")  # GPU

dpt_X_train = dpt.asarray(X_train, usm_type="device", sycl_queue=q)
dpt_y_train = dpt.asarray(y_train, usm_type="device", sycl_queue=q)
dpt_X_test = dpt.asarray(X_test, usm_type="device", sycl_queue=q)
dpt_y_test = dpt.asarray(y_test, usm_type="device", sycl_queue=q)


rf = RandomForestClassifier(max_depth=2, random_state=0).fit(dpt_X_train, dpt_y_train)

pred = rf.predict(dpt_X_test)

print(accuracy_score(dpt.to_numpy(pred), dpt.to_numpy(dpt_y_test)))
print("Random Forest classification results:")
print("Ground truth (first 5 observations on rank {}):\n{}".format(mpi_rank, y_test[:5]))
print("Classification results (first 5 observations on rank {}):\n{}"
      .format(mpi_rank, dpt.to_numpy(pred)[:5]))
print("Accuracy for entire rank {} (256 classes): {}"
      .format(mpi_rank, accuracy_score(y_test, dpt.to_numpy(pred))))
