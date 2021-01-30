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

# daal4py outlier detection bacon example for shared memory systems

import daal4py as d4p
import numpy as np

# let's try to use pandas' fast csv reader
try:
    import pandas

    def read_csv(f, c, t=np.float64):
        return pandas.read_csv(f, usecols=c, delimiter=',', header=None, dtype=t)
except ImportError:
    # fall back to numpy loadtxt
    def read_csv(f, c, t=np.float64):
        return np.loadtxt(f, usecols=c, delimiter=',', ndmin=2)


def main(readcsv=read_csv, method='defaultDense'):
    # Input file
    infile = "./data/batch/outlierdetection.csv"

    # Retrieve the data from the input file
    data = readcsv(infile, range(3))

    # Create an algorithm to detect outliers using the default method
    algorithm = d4p.bacon_outlier_detection()

    # Compute outliers and get the computed results
    res = algorithm.compute(data)

    # result provides weights
    assert res.weights.shape == (data.shape[0], 1)

    return (data, res)


if __name__ == "__main__":
    (data, res) = main()

    print("\nInput data\n", data)
    print("\nOutlier detection result (Bacon method) weights:\n", res.weights)
    print('All looks good!')
