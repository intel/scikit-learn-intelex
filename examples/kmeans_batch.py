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

# daal4py K-Means example for shared memory systems

import daal4py as d4p
from numpy import loadtxt, allclose

if __name__ == "__main__":

    infile = "./data/batch/kmeans_dense.csv"
    method = 'svdDense'
    nClusters = 10
    maxIter = 25

    # configure a kmeans-init using mt19937 RNG
    engine = d4p.engines_mt19937()
    initrain_algo = d4p.kmeans_init(nClusters, method="randomDense", engine=engine)
    # Load the data
    data = loadtxt(infile, delimiter=',')
    # compute initial centroids
    initrain_result = initrain_algo.compute(data)
    # The results provides the initial centroids
    assert initrain_result.centroids.shape[0] == nClusters

    # configure kmeans main object: we also request the cluster assignments
    algo = d4p.kmeans(nClusters, maxIter, assignFlag=True)
    # compute the clusters/centroids
    result = algo.compute(data, initrain_result.centroids)
    
    # Note: we could have done this in just one line:
    # d4p.kmeans(nClusters, maxIter, assignFlag=True).compute(data, d4p.kmeans_init(nClusters, method="plusPlusDense").compute(data).centroids)

    # Kmeans result objects provide assignments (if requested), centroids, goalFunction, nIterations and objectiveFunction
    assert result.centroids.shape[0] == nClusters
    assert result.assignments.shape == (data.shape[0], 1)
    assert result.nIterations <= maxIter

    print("\nFirst 10 cluster assignments:\n", result.assignments[0:10])
    print("\nFirst 10 dimensions of centroids:\n", result.centroids[:,0:10])
    print("\nObjective function value:\n", result.objectiveFunction)
    print('All looks good!')
