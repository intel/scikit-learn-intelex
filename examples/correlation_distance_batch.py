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

# daal4py correlation distance example for shared memory systems

import daal4py as d4p
import numpy as np
import os

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
    data = readcsv(os.path.join('data', 'batch', 'distance.csv'), range(10))

    # Create algorithm to compute correlation distance (no parameters)
    algorithm = d4p.correlation_distance()

    # Computed correlation distance with file or numpy array
    res1 = algorithm.compute(os.path.join('data', 'batch', 'distance.csv'))
    res2 = algorithm.compute(data)

    assert np.allclose(res1.correlationDistance, res2.correlationDistance)

    return res1


if __name__ == "__main__":
    res = main()
    print(
        "\nCorrelation distance (first 15 rows/columns):\n",
        res.correlationDistance[0:15, 0:15]
    )
    print("All looks good!")
