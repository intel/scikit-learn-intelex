#*******************************************************************************
# Copyright 2014-2020 Intel Corporation
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

# daal4py DBSCAN example for shared memory systems

import daal4py as d4p
import numpy as np
import os
from daal4py.oneapi import sycl_context, sycl_buffer

# let's try to use pandas' fast csv reader
try:
    import pandas
    read_csv = lambda f, c=None, t=np.float64: pandas.read_csv(f, usecols=c, delimiter=',', header=None, dtype=t)
except:
    # fall back to numpy loadtxt
    read_csv = lambda f, c=None, t=np.float64: np.loadtxt(f, usecols=c, delimiter=',', ndmin=2)


# Commone code for both CPU and GPU computations
def compute(data, minObservations, epsilon):
    # configure dbscan main object: we also request the indices and observations of cluster cores
    algo = d4p.dbscan(minObservations=minObservations, epsilon=epsilon, resultsToCompute='computeCoreIndices|computeCoreObservations', memorySavingMode=1)
    return algo.compute(data)


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


def main(readcsv=read_csv, method='defaultDense'):
    infile = os.path.join('..', 'data', 'batch', 'dbscan_dense.csv')
    epsilon = 0.02
    minObservations = 180

    # Load the data
    data = readcsv(infile)

    # Using of the classic way (computations on CPU)
    result_classic = compute(data, minObservations, epsilon)

    data = to_numpy(data)

    # It is possible to specify to make the computations on GPU
    with sycl_context('gpu'):
        sycl_data = sycl_buffer(data)
        result_gpu = compute(sycl_data, minObservations, epsilon)

    # PCA result objects provide eigenvalues, eigenvectors, means and variances
    assert result_classic.assignments.shape == (data.shape[0], 1)
    assert result_classic.coreObservations.shape == (result_classic.coreIndices.shape[0], data.shape[1])

    assert np.allclose(result_classic.assignments, result_gpu.assignments)
    assert np.allclose(result_classic.coreIndices, result_gpu.coreIndices)
    assert np.allclose(result_classic.coreObservations, result_gpu.coreObservations)
    assert np.allclose(result_classic.nClusters, result_gpu.nClusters)

    return result_classic


if __name__ == "__main__":
    result = main()
    print("\nFirst 10 cluster assignments:\n", result.assignments[0:10])
    print("\nFirst 10 cluster core indices:\n", result.coreIndices[0:10])
    print("\nFirst 10 cluster core observations:\n", result.coreObservations[0:10])
    print("\nNumber of clusters:\n", result.nClusters)
    print('All looks good!')
