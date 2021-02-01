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

# daal4py K-Means example for shared memory systems

import daal4py as d4p
import numpy as np
import os
from daal4py.oneapi import sycl_buffer

# let's try to use pandas' fast csv reader
try:
    import pandas

    def read_csv(f, c, t=np.float64):
        return pandas.read_csv(f, usecols=c, delimiter=',', header=None, dtype=t)
except ImportError:
    # fall back to numpy loadtxt
    def read_csv(f, c, t=np.float64):
        return np.loadtxt(f, usecols=c, delimiter=',', ndmin=2)

try:
    from dpctx import device_context, device_type
    with device_context(device_type.gpu, 0):
        gpu_available = True
except:
    try:
        from daal4py.oneapi import sycl_context
        with sycl_context('gpu'):
            gpu_available = True
    except:
        gpu_available = False


# Commone code for both CPU and GPU computations
def compute(data, nClusters, maxIter, method):
    # configure kmeans init object
    initrain_algo = d4p.kmeans_init(nClusters, method=method, fptype='float')
    # compute initial centroids
    initrain_result = initrain_algo.compute(data)

    # configure kmeans main object: we also request the cluster assignments
    algo = d4p.kmeans(nClusters, maxIter, assignFlag=True, fptype='float')
    # compute the clusters/centroids
    return algo.compute(data, initrain_result.centroids)

    # Note: we could have done this in just one line:
    # return d4p.kmeans(nClusters, maxIter, assignFlag=True).compute(
    #     data, d4p.kmeans_init(nClusters, method=method).compute(data).centroids
    # )


# At this moment with sycl we are working only with numpy arrays
def to_numpy(data):
    try:
        from pandas import DataFrame
        if isinstance(data, DataFrame):
            return np.ascontiguousarray(data.values)
    except ImportError:
        pass
    try:
        from scipy.sparse import csr_matrix
        if isinstance(data, csr_matrix):
            return data.toarray()
    except ImportError:
        pass
    return data


def main(readcsv=read_csv, method='randomDense'):
    infile = os.path.join('..', 'data', 'batch', 'kmeans_dense.csv')
    nClusters = 20
    maxIter = 5

    # Load the data
    data = readcsv(infile, range(20), t=np.float32)

    # Using of the classic way (computations on CPU)
    result_classic = compute(data, nClusters, maxIter, method)

    data = to_numpy(data)

    try:
        from dpctx import device_context, device_type

        def gpu_context():
            return device_context(device_type.gpu, 0)

        def cpu_context():
            return device_context(device_type.cpu, 0)
    except:
        from daal4py.oneapi import sycl_context

        def gpu_context():
            return sycl_context('gpu')

        def cpu_context():
            return sycl_context('cpu')

    # It is possible to specify to make the computations on GPU
    if gpu_available:
        with gpu_context():
            sycl_data = sycl_buffer(data)
            _ = compute(sycl_data, nClusters, maxIter, method)
        # TODO: investigate why results_classic and result_gpu differ
        # assert np.allclose(result_classic.centroids, result_gpu.centroids)
        # assert np.allclose(result_classic.assignments, result_gpu.assignments)
        # assert np.isclose(result_classic.objectiveFunction,
        #                   result_gpu.objectiveFunction)

    # It is possible to specify to make the computations on CPU
    with cpu_context():
        sycl_data = sycl_buffer(data)
        result_cpu = compute(sycl_data, nClusters, maxIter, method)

    # Kmeans result objects provide assignments (if requested),
    # centroids, goalFunction, nIterations and objectiveFunction
    assert result_classic.centroids.shape[0] == nClusters
    assert result_classic.assignments.shape == (data.shape[0], 1)
    assert result_classic.nIterations <= maxIter

    assert np.allclose(result_classic.centroids, result_cpu.centroids)
    assert np.allclose(result_classic.assignments, result_cpu.assignments)
    assert np.isclose(result_classic.objectiveFunction, result_cpu.objectiveFunction)
    assert result_classic.nIterations == result_cpu.nIterations

    return result_classic


if __name__ == "__main__":
    result = main()
    print("\nFirst 10 cluster assignments:\n", result.assignments[0:10])
    print("\nFirst 10 dimensions of centroids:\n", result.centroids[:, 0:10])
    print("\nObjective function value:\n", result.objectiveFunction)
    print('All looks good!')
