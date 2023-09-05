# ==============================================================================
# Copyright 2014 Intel Corporation
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
# ==============================================================================

# daal4py low order moments example for shared memory systems

import os

import numpy as np

import daal4py as d4p
from daal4py.oneapi import sycl_buffer

# let's try to use pandas' fast csv reader
try:
    import pandas

    def read_csv(f, c, t=np.float64):
        return pandas.read_csv(f, usecols=c, delimiter=",", header=None, dtype=t)

except ImportError:
    # fall back to numpy loadtxt
    def read_csv(f, c, t=np.float64):
        return np.loadtxt(f, usecols=c, delimiter=",", ndmin=2)


try:
    from daal4py.oneapi import sycl_context

    with sycl_context("gpu"):
        gpu_available = True
except:
    gpu_available = False


# Commone code for both CPU and GPU computations
def compute(data, method):
    alg = d4p.low_order_moments(method=method, fptype="float")
    return alg.compute(data)


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


def main(readcsv=read_csv, method="defaultDense"):
    # read data from file
    file = os.path.join("..", "data", "batch", "covcormoments_dense.csv")
    data = readcsv(file, range(10), t=np.float32)

    # Using of the classic way (computations on CPU)
    result_classic = compute(data, method)

    data = to_numpy(data)

    # It is possible to specify to make the computations on GPU
    if gpu_available:
        with sycl_context("gpu"):
            sycl_data = sycl_buffer(data)
            result_gpu = compute(sycl_data, "defaultDense")
        for name in [
            "minimum",
            "maximum",
            "sum",
            "sumSquares",
            "sumSquaresCentered",
            "mean",
            "secondOrderRawMoment",
            "variance",
            "standardDeviation",
            "variation",
        ]:
            assert np.allclose(getattr(result_classic, name), getattr(result_gpu, name))

    # It is possible to specify to make the computations on CPU
    with sycl_context("cpu"):
        sycl_data = sycl_buffer(data)
        result_cpu = compute(sycl_data, "defaultDense")

    # result provides minimum, maximum, sum, sumSquares, sumSquaresCentered,
    # mean, secondOrderRawMoment, variance, standardDeviation, variation
    assert all(
        getattr(result_classic, name).shape == (1, data.shape[1])
        for name in [
            "minimum",
            "maximum",
            "sum",
            "sumSquares",
            "sumSquaresCentered",
            "mean",
            "secondOrderRawMoment",
            "variance",
            "standardDeviation",
            "variation",
        ]
    )

    for name in [
        "minimum",
        "maximum",
        "sum",
        "sumSquares",
        "sumSquaresCentered",
        "mean",
        "secondOrderRawMoment",
        "variance",
        "standardDeviation",
        "variation",
    ]:
        assert np.allclose(getattr(result_classic, name), getattr(result_cpu, name))

    return result_classic


if __name__ == "__main__":
    res = main()
    # print results
    print("\nMinimum:\n", res.minimum)
    print("\nMaximum:\n", res.maximum)
    print("\nSum:\n", res.sum)
    print("\nSum of squares:\n", res.sumSquares)
    print("\nSum of squared difference from the means:\n", res.sumSquaresCentered)
    print("\nMean:\n", res.mean)
    print("\nSecond order raw moment:\n", res.secondOrderRawMoment)
    print("\nVariance:\n", res.variance)
    print("\nStandard deviation:\n", res.standardDeviation)
    print("\nVariation:\n", res.variation)
    print("All looks good!")
