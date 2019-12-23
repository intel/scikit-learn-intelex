#*******************************************************************************
# Copyright 2014-2019 Intel Corporation
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
import numpy as np
from daal4py.oneapi import sycl_context, sycl_buffer

# let's try to use pandas' fast csv reader
try:
    import pandas
    read_csv = lambda f, c, t=np.float64: pandas.read_csv(f, usecols=c, delimiter=',', header=None, dtype=t)
except:
    # fall back to numpy loadtxt
    read_csv = lambda f, c, t=np.float64: np.loadtxt(f, usecols=c, delimiter=',', ndmin=2)


# Commone code for both CPU and GPU computations
def compute(data, nClusters, maxIter, method):
    # configure kmeans init object
    initrain_algo = d4p.kmeans_init(nClusters, method=method)
    # compute initial centroids
    initrain_result = initrain_algo.compute(data)

    # configure kmeans main object: we also request the cluster assignments
    algo = d4p.kmeans(nClusters, maxIter, assignFlag=True)
    # compute the clusters/centroids
    return algo.compute(data, initrain_result.centroids)

    # Note: we could have done this in just one line:
    # return d4p.kmeans(nClusters, maxIter, assignFlag=True).compute(data, d4p.kmeans_init(nClusters, method=method).compute(data).centroids)


# At this moment with sycl we are working only with numpy arrays
def to_numpy(data):
    try:
        from pandas import DataFrame
        if isinstance(data, DataFrame):
            return np.ascontiguousarray(data.values)
    except:
        pass
    try:
        from scipy.sparse import csr_matrix
        if isinstance(data, csr_matrix):
            return data.toarray()
    except:
        pass
    return data


def main(readcsv=read_csv, method='randomDense'):
    infile = "./data/batch/kmeans_dense.csv"
    nClusters = 20
    maxIter = 5

    # Load the data
    data = readcsv(infile, range(20))

    # Using of the classic way (computations on CPU)
    result_classic = compute(data, nClusters, maxIter, method)

    data = to_numpy(data)

    # It is possible to specify to make the computations on GPU
    with sycl_context('gpu'):
        sycl_data = sycl_buffer(data)
        result_gpu = compute(sycl_data, nClusters, maxIter, method)

    # Kmeans result objects provide assignments (if requested), centroids, goalFunction, nIterations and objectiveFunction
    assert result_classic.centroids.shape[0] == nClusters
    assert result_classic.assignments.shape == (data.shape[0], 1)
    assert result_classic.nIterations <= maxIter

    assert np.allclose(result_classic.centroids, result_gpu.centroids)
    assert np.allclose(result_classic.assignments, result_gpu.assignments)
    assert np.isclose(result_classic.objectiveFunction, result_gpu.objectiveFunction)
    assert result_classic.nIterations == result_gpu.nIterations

    return result_classic


if __name__ == "__main__":
    result = main()
    print("\nFirst 10 cluster assignments:\n", result.assignments[0:10])
    print("\nFirst 10 dimensions of centroids:\n", result.centroids[:,0:10])
    print("\nObjective function value:\n", result.objectiveFunction)
    print('All looks good!')
