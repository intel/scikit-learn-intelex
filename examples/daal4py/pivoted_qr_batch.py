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

# daal4py pivoted QR example for shared memory systems

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

    # configure a pivoted QR object
    algo = d4p.pivoted_qr()

    # let's provide a file directly, not a table/array
    result1 = algo.compute(infile)

    # We can also load the data ourselfs and provide the numpy array
    data = readcsv(infile)
    _ = algo.compute(data)

    # pivoted QR result objects provide matrixQ, matrixR and permutationMatrix
    return result1


if __name__ == "__main__":
    result = main()
    print("Orthogonal matrix Q (:10):\n", result.matrixQ[:10])
    print("Triangular matrix R:\n", result.matrixR)
    print("\nPermutation matrix P:\n", result.permutationMatrix)
    print('All looks good!')
