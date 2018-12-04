#*******************************************************************************
# Copyright 2014-2018 Intel Corporation
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

# daal4py cholesky example for shared memory systems

import daal4py as d4p
import numpy as np

# let's try to use pandas' fast csv reader
try:
    import pandas
    read_csv = lambda f, c: pandas.read_csv(f, usecols=c, delimiter=',', header=None, dtype=np.float64)
except:
    # fall back to numpy loadtxt
    read_csv = lambda f, c: np.loadtxt(f, usecols=c, delimiter=',', ndmin=2)


def main():
    infile = "./data/batch/cholesky.csv"

    # configure a cholesky object
    algo = d4p.cholesky()
    
    # let's provide a file directly, not a table/array
    return algo.compute(infile)
    # cholesky result objects provide choleskyFactor


if __name__ == "__main__":
    result = main()
    print("\nFactor:\n", result.choleskyFactor)
    print('All looks good!')
    np.savetxt('c.csv', result.choleskyFactor, delimiter=',')
