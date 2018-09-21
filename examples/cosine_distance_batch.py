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

# daal4py cosine distance example for shared memory systems

import daal4py as d4p
import numpy as np
import os

# let's try to use pandas' fast csv reader
try:
    import pandas
    read_csv = lambda f, c: pandas.read_csv(f, usecols=c, delimiter=',', header=None).values
except:
    # fall back to numpy loadtxt
    read_csv = lambda f, c: np.loadtxt(f, usecols=c, delimiter=',')


def main():
    # Create algorithm to compute cosine distance (no parameters)
    algorithm = d4p.cosine_distance()

    # Computed cosine distance
    return algorithm.compute(os.path.join('data', 'batch', 'distance.csv'))


if __name__ == "__main__":
    res = main()
    np.set_printoptions(threshold=np.nan)
    np.set_printoptions(linewidth=np.nan)
    print("\nCosine distance (first 15 rows/columns):\n", res.cosineDistance[0:15,0:15])
    print("All looks good!")
