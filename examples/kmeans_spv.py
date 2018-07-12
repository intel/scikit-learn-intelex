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

# daal4py K-Means example for distributed memory systems; Single Process View mode
# run like this:
#    mpirun -genv DIST_CNC=MPI -n 4 python ./kmeans_spmd.py

import daal4py as d4p
from numpy import loadtxt, allclose

if __name__ == "__main__":

    # Initialize SPV mode
    d4p.daalinit()

    infile = "./data/distributed/kmeans_dense.csv"
    method = 'svdDense'
    nClusters = 10
    maxIter = 25

    # configure a kmeans-init
    initrain_algo = d4p.kmeans_init(nClusters, method="plusPlusDense", distributed=True)
    # Load the data
    data = loadtxt(infile, delimiter=',')
    # We need partioned input data, let's slice the data
    rpp = int(data.shape[0]/d4p.num_procs())
    data = [data[rpp*x:rpp*x+rpp,:] for x in range(d4p.num_procs())]
    # Note, providing a list of files instead also distributes the file read!

    # compute initial centroids
    initrain_result = initrain_algo.compute(data)
    # The results provides the initial centroids
    assert initrain_result.centroids.shape[0] == nClusters

    # configure kmeans main object
    algo = d4p.kmeans(nClusters, maxIter, distributed=True)
    # compute the clusters/centroids
    result = algo.compute(data, initrain_result.centroids)
    
    # Note: we could have done this in just one line:
    # d4p.kmeans(nClusters, maxIter, assignFlag=True, distributed=True).compute(data, d4p.kmeans_init(nClusters, method="plusPlusDense", distributed=True).compute(data).centroids)

    # Kmeans result objects provide assignments (if requested), centroids, goalFunction, nIterations and objectiveFunction
    assert result.centroids.shape[0] == nClusters
    # we'd need an extra call to kmeans.compute(10, 0) to get the assignments; getting assignments is not yet supported in dist mode
    assert result.assignments == None
    assert result.nIterations <= maxIter

    print('All looks good!')
    d4p.daalfini()
