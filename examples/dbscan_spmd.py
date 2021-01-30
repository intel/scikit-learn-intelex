#===============================================================================
# Copyright 2014-2021 Intel Corporation
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

# daal4py DBSCAN example for distributed memory systems; SPMD mode
# run like this:
#    mpirun -n 4 python ./dbscan_spmd.py

import daal4py as d4p
import numpy as np


def main(method='defaultDense'):
    infile = "./data/batch/dbscan_dense.csv"
    epsilon = 0.04
    minObservations = 45

    # Load the data
    data = np.loadtxt(infile, delimiter=',')
    rpp = int(data.shape[0] / d4p.num_procs())
    data = data[rpp * d4p.my_procid(): rpp * d4p.my_procid() + rpp, :]

    # configure dbscan main object
    algo = d4p.dbscan(minObservations=minObservations, epsilon=epsilon, distributed=True)
    # and compute
    result = algo.compute(data)

    return result


if __name__ == "__main__":
    # Initialize SPMD mode
    d4p.daalinit()
    result = main()
    print("\nResults on node with id = ", d4p.my_procid(), " :\n",
          "\nFirst 10 cluster assignments:\n", result.assignments[0:10],
          "\nNumber of clusters:\n", result.nClusters)
    d4p.daalfini()
