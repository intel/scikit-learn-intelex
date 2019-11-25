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

# daal4py PCA example for shared memory systems

import daal4py as d4p
import numpy as np
from daal4py.oneapi import sycl_context, sycl_buffer

# let's try to use pandas' fast csv reader
try:
    import pandas
    read_csv = lambda f, c=None, t=np.float64: pandas.read_csv(f, usecols=c, delimiter=',', header=None, dtype=t)
except:
    # fall back to numpy loadtxt
    read_csv = lambda f, c=None, t=np.float64: np.loadtxt(f, usecols=c, delimiter=',', ndmin=2)


# Commone code for both CPU and GPU computations
def compute(data):
    # 'normalization' is an optional parameter to PCA; we use z-score which could be configured differently
    zscore = d4p.normalization_zscore()
    # configure a PCA object
    algo = d4p.pca(resultsToCompute="mean|variance|eigenvalue", isDeterministic=True, normalization=zscore)
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


def main(readcsv=read_csv, method='svdDense'):
    infile = "./data/batch/pca_normalized.csv"

    # Load the data
    data = readcsv(infile)

    # Using of the classic way (computations on CPU)
    result_classic = compute(data)
    
    data = to_numpy(data)

    # It is possible to specify to make the computations on GPU
    with sycl_context('gpu'):
        sycl_data = sycl_buffer(data)
        result_gpu = compute(sycl_data)

    # It is possible to specify to make the computations on CPU
    with sycl_context('cpu'):
        sycl_data = sycl_buffer(data)
        result_cpu = compute(sycl_data)

    # PCA result objects provide eigenvalues, eigenvectors, means and variances
    assert result_classic.eigenvalues.shape == (1, data.shape[1])
    assert result_classic.eigenvectors.shape == (data.shape[1], data.shape[1])
    assert result_classic.means.shape == (1, data.shape[1])
    assert result_classic.variances.shape == (1, data.shape[1])

    assert np.allclose(result_classic.eigenvalues, result_gpu.eigenvalues)
    assert np.allclose(result_classic.eigenvectors, result_gpu.eigenvectors)
    assert np.allclose(result_classic.means, result_gpu.means, atol=1e-7)
    assert np.allclose(result_classic.variances, result_gpu.variances)

    assert np.allclose(result_classic.eigenvalues, result_cpu.eigenvalues)
    assert np.allclose(result_classic.eigenvectors, result_cpu.eigenvectors)
    assert np.allclose(result_classic.means, result_cpu.means, atol=1e-7)
    assert np.allclose(result_classic.variances, result_cpu.variances)

    return result_classic


if __name__ == "__main__":
    result = main()
    print("\nEigenvalues:\n", result.eigenvalues)
    print("\nEigenvectors:\n", result.eigenvectors)
    print("\nMeans:\n", result.means)
    print("\nVariances:\n", result.variances)
    print('All looks good!')
