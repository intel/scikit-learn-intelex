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

# daal4py QR example for shared memory systems

import daal4py as d4p
import numpy as np

# let's try to use pandas' fast csv reader
try:
    import pandas

    def read_csv(f, c=None, t=np.float64):
        return pandas.read_csv(f, usecols=c, delimiter=',', header=None, dtype=t)
except ImportError:
    # fall back to numpy loadtxt
    def read_csv(f, c=None, t=np.float64):
        return np.loadtxt(f, usecols=c, delimiter=',', ndmin=2)


def main(readcsv=read_csv, method='svdDense'):
    infile = "./data/batch/qr.csv"

    # configure a QR object
    algo = d4p.qr()

    # let's provide a file directly, not a table/array
    result1 = algo.compute(infile)

    # We can also load the data ourselfs and provide the numpy array
    data = readcsv(infile)
    result2 = algo.compute(data)

    # QR result provide matrixQ and matrixR
    assert result1.matrixQ.shape == data.shape
    assert result1.matrixR.shape == (data.shape[1], data.shape[1])

    assert np.allclose(result1.matrixQ, result2.matrixQ, atol=1e-07)
    assert np.allclose(result1.matrixR, result2.matrixR, atol=1e-07)

    if hasattr(data, 'toarray'):
        data = data.toarray()  # to make the next assertion work with scipy's csr_matrix
    assert np.allclose(data, np.matmul(result1.matrixQ, result1.matrixR))

    return data, result1


if __name__ == "__main__":
    (_, result) = main()
    print(result)
    print('All looks good!')
