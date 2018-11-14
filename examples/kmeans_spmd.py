#*******************************************************************************
# Copyright 2014-2018 Intel Corporation
# All Rights Reserved.
#
# This software is licensed under the Apache License, Version 2.0 (the
# "License"), the following terms apply:
#
# You may not use this file except in compliance with the License.  You may
# obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# See the License for the specific language governing permissions and
# limitations under the License.
#*******************************************************************************

# daal4py K-Means example for distributed memory systems; SPMD mode
# run like this:
#    mpirun -n 4 python ./kmeans_spmd.py

import daal4py as d4p
from numpy import loadtxt, allclose

if __name__ == "__main__":

    # Initialize SPMD mode
    d4p.daalinit()

    infile = "./data/distributed/kmeans_dense.csv"
    nClusters = 10
    maxIter = 25

    # configure a kmeans-init
    init_algo = d4p.kmeans_init(nClusters, method="plusPlusDense", distributed=True)
    # Load the data
    data = loadtxt(infile, delimiter=',')
    # now slice the data, it would have been better to read only what we need, of course...
    rpp = int(data.shape[0]/d4p.num_procs())
    data = data[rpp*d4p.my_procid():rpp*d4p.my_procid()+rpp,:]

    # compute initial centroids
    init_result = init_algo.compute(data)
    # The results provides the initial centroids
    assert init_result.centroids.shape[0] == nClusters

    # configure kmeans main object
    algo = d4p.kmeans(nClusters, maxIter, distributed=True)
    # compute the clusters/centroids
    result = algo.compute(data, init_result.centroids)
    
    # Note: we could have done this in just one line:
    # d4p.kmeans(nClusters, maxIter, assignFlag=True, distributed=True).compute(data, d4p.kmeans_init(nClusters, method="plusPlusDense", distributed=True).compute(data).centroids)

    # Kmeans result objects provide assignments (if requested), centroids, goalFunction, nIterations and objectiveFunction
    assert result.centroids.shape[0] == nClusters
    print(result.nIterations, result.centroids[0], maxIter)
    # we'd need an extra call to kmeans.compute(10, 0) to get the assignments; getting assignments is not yet supported in dist mode
    assert result.assignments == None
    assert result.nIterations <= maxIter

    print('All looks good!')
    d4p.daalfini()
