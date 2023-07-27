#===============================================================================
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
#===============================================================================

# daal4py outlier detection univariate example for shared memory systems

import numpy as np

import daal4py as d4p

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

    # Create an algorithm to detect outliers (univariate)
    algorithm = d4p.univariate_outlier_detection()

    # Compute outliers and get the computed results
    res = algorithm.compute(data, None, None, None)

    # result provides weights
    assert res.weights.shape == (data.shape[0], 3)

    return (data, res)


if __name__ == "__main__":
    (data, res) = main()

    print("\nInput data\n", data)
    print("\nOutlier detection result (univariate) weights:\n", res.weights)
    print('All looks good!')
