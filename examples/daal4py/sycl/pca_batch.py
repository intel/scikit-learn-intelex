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

# daal4py PCA example for shared memory systems

import daal4py as d4p
import numpy as np
import os
from daal4py.oneapi import sycl_buffer

# let's try to use pandas' fast csv reader
try:
    import pandas

    def read_csv(f, c=None, t=np.float64):
        return pandas.read_csv(f, usecols=c, delimiter=',', header=None, dtype=t)
except ImportError:
    # fall back to numpy loadtxt
    def read_csv(f, c=None, t=np.float64):
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
def compute(data):
    # 'normalization' is an optional parameter to PCA;
    # we use z-score which could be configured differently
    zscore = d4p.normalization_zscore()
    # configure a PCA object
    algo = d4p.pca(resultsToCompute="mean|variance|eigenvalue",
                   isDeterministic=True, normalization=zscore)
    return algo.compute(data)


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


def main(readcsv=read_csv, method='svdDense'):
    infile = os.path.join('..', 'data', 'batch', 'pca_normalized.csv')

    # Load the data
    data = readcsv(infile, t=np.float32)

    # Using of the classic way (computations on CPU)
    result_classic = compute(data)

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
            result_gpu = compute(sycl_data)
        assert np.allclose(result_classic.eigenvalues, result_gpu.eigenvalues)
        assert np.allclose(result_classic.eigenvectors, result_gpu.eigenvectors)
        assert np.allclose(result_classic.means, result_gpu.means, atol=1e-7)
        assert np.allclose(result_classic.variances, result_gpu.variances)

    # It is possible to specify to make the computations on CPU
    with cpu_context():
        sycl_data = sycl_buffer(data)
        result_cpu = compute(sycl_data)

    # PCA result objects provide eigenvalues, eigenvectors, means and variances
    assert result_classic.eigenvalues.shape == (1, data.shape[1])
    assert result_classic.eigenvectors.shape == (data.shape[1], data.shape[1])
    assert result_classic.means.shape == (1, data.shape[1])
    assert result_classic.variances.shape == (1, data.shape[1])

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
