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

# daal4py covariance example for streaming on shared memory systems

import os

# let's use a generator for getting stream from file (defined in stream.py)
import sys
from pathlib import Path

import numpy as np

import daal4py as d4p
from daal4py.oneapi import sycl_buffer

sys.path.insert(0, "..")

try:
    from daal4py.oneapi import sycl_context

    with sycl_context("gpu"):
        gpu_available = True
except Exception:
    gpu_available = False

try:
    import pandas

    def read_csv(f, c=None, s=0, n=None, t=np.float64):
        return pandas.read_csv(
            f, usecols=c, delimiter=",", header=None, skiprows=s, nrows=n, dtype=t
        )

except Exception:
    # fall back to numpy genfromtxt
    def read_csv(f, c=None, s=0, n=np.iinfo(np.int64).max):
        a = np.genfromtxt(f, usecols=c, delimiter=",", skip_header=s, max_rows=n)
        if a.shape[0] == 0:
            raise Exception("done")
        if a.ndim == 1:
            return a[:, np.newaxis]
        return a


# a generator which reads a file in chunks
def read_next(file, chunksize, readcsv=read_csv):
    assert os.path.isfile(file)
    s = 0
    while True:
        # if found a smaller chunk we set s to < 0 to indicate eof
        if s < 0:
            return
        a = read_csv(file, s=s, n=chunksize)
        # last chunk is usually smaller, if not,
        # numpy will print warning in next iteration
        if chunksize > a.shape[0]:
            s = -1
        else:
            s += a.shape[0]
        yield a


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


def main(readcsv=None, method="defaultDense"):
    data_path = Path(__file__).parent.parent / "data" / "batch"
    infile = data_path / "covcormoments_dense.csv"

    # Using of the classic way (computations on CPU)
    # configure a covariance object
    algo = d4p.covariance(streaming=True, fptype="float")
    # get the generator (defined in stream.py)...
    rn = read_next(infile, 112, readcsv)
    # ... and iterate through chunks/stream
    for chunk in rn:
        algo.compute(chunk)
    # finalize computation
    result_classic = algo.finalize()

    # It is possible to specify to make the computations on GPU
    if gpu_available:
        with sycl_context("gpu"):
            # configure a covariance object
            algo = d4p.covariance(streaming=True, fptype="float")
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

    return result_classic


if __name__ == "__main__":
    res = main()
    print("Covariance matrix:\n", res.covariance)
    print("Mean vector:\n", res.mean)
    print("All looks good!")
