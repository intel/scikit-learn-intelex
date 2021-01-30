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

# daal4py covariance example for streaming on shared memory systems

import daal4py as d4p
import numpy as np
import os
from daal4py.oneapi import sycl_buffer

# let's use a generator for getting stream from file (defined in stream.py)
import sys
sys.path.insert(0, '..')
from stream import read_next

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


def main(readcsv=None, method='defaultDense'):
    infile = os.path.join('..', 'data', 'batch', 'covcormoments_dense.csv')

    # Using of the classic way (computations on CPU)
    # configure a covariance object
    algo = d4p.covariance(streaming=True, fptype='float')
    # get the generator (defined in stream.py)...
    rn = read_next(infile, 112, readcsv)
    # ... and iterate through chunks/stream
    for chunk in rn:
        algo.compute(chunk)
    # finalize computation
    result_classic = algo.finalize()

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
            # configure a covariance object
            algo = d4p.covariance(streaming=True, fptype='float')
            # get the generator (defined in stream.py)...
            rn = read_next(infile, 112, readcsv)
            # ... and iterate through chunks/stream
            for chunk in rn:
                sycl_chunk = sycl_buffer(to_numpy(chunk))
                algo.compute(sycl_chunk)
            # finalize computation
            result_gpu = algo.finalize()
        assert np.allclose(result_classic.covariance, result_gpu.covariance)
        assert np.allclose(result_classic.mean, result_gpu.mean)
        assert np.allclose(result_classic.correlation, result_gpu.correlation)

    # It is possible to specify to make the computations on CPU
    with cpu_context():
        # configure a covariance object
        algo = d4p.covariance(streaming=True, fptype='float')
        # get the generator (defined in stream.py)...
        rn = read_next(infile, 112, readcsv)
        # ... and iterate through chunks/stream
        for chunk in rn:
            sycl_chunk = sycl_buffer(to_numpy(chunk))
            algo.compute(sycl_chunk)
        # finalize computation
        result_cpu = algo.finalize()

    # covariance result objects provide correlation, covariance and mean

    assert np.allclose(result_classic.covariance, result_cpu.covariance)
    assert np.allclose(result_classic.mean, result_cpu.mean)
    assert np.allclose(result_classic.correlation, result_cpu.correlation)

    return result_classic


if __name__ == "__main__":
    res = main()
    print("Covariance matrix:\n", res.covariance)
    print("Mean vector:\n", res.mean)
    print('All looks good!')
