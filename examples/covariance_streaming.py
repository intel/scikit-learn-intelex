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

# let's use a generator for getting stream from file (defined in stream.py)
from stream import read_next


def main(readcsv=None, method='defaultDense'):
    infile = "./data/batch/covcormoments_dense.csv"

    # configure a covariance object
    algo = d4p.covariance(streaming=True)

    # get the generator (defined in stream.py)...
    rn = read_next(infile, 112, readcsv)
    # ... and iterate through chunks/stream
    for chunk in rn:
        algo.compute(chunk)

    # finalize computation
    result = algo.finalize()

    return result


if __name__ == "__main__":
    res = main()
    print("Covariance matrix:\n", res.covariance)
    print("Mean vector:\n", res.mean)
    print('All looks good!')
