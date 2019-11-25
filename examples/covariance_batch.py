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

# daal4py covariance example for shared memory systems

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
def compute(data, method):
    # configure a covariance object
    algo = d4p.covariance(method=method)
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
    infile = "./data/batch/covcormoments_dense.csv"

    # Load the data
    data = readcsv(infile, range(10))

    # Using of the classic way (computations on CPU)
    result_classic = compute(data, method)
    
    data = to_numpy(data)

    # It is possible to specify to make the computations on GPU
    with sycl_context('gpu'):
        sycl_data = sycl_buffer(data)
        result_gpu = compute(sycl_data, 'defaultDense')

    # It is possible to specify to make the computations on CPU
    with sycl_context('cpu'):
        sycl_data = sycl_buffer(data)
        result_cpu = compute(sycl_data, 'defaultDense')

    # covariance result objects provide correlation, covariance and mean
    assert np.allclose(result_classic.covariance, result_gpu.covariance)
    assert np.allclose(result_classic.mean, result_gpu.mean)
    assert np.allclose(result_classic.correlation, result_gpu.correlation)

    assert np.allclose(result_classic.covariance, result_cpu.covariance)
    assert np.allclose(result_classic.mean, result_cpu.mean)
    assert np.allclose(result_classic.correlation, result_cpu.correlation)

    return result_classic


if __name__ == "__main__":
    res = main()
    print("Covariance matrix:\n", res.covariance)
    print("Mean vector:\n", res.mean)
    print('All looks good!')
