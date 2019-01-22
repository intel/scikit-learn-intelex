#*******************************************************************************
# Copyright 2014-2019 Intel Corporation
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

# daal4py em_gmm example for shared memory systems

import daal4py as d4p
import numpy as np

# let's try to use pandas' fast csv reader
try:
    import pandas
    read_csv = lambda f, c=None, t=np.float64: pandas.read_csv(f, usecols=c, delimiter=',', header=None, dtype=t)
except:
    # fall back to numpy loadtxt
    read_csv = lambda f, c=None, t=np.float64: np.loadtxt(f, usecols=c, delimiter=',', ndmin=2)


def main(readcsv=read_csv, method='defaultDense'):
    nComponents = 2
    infile = "./data/batch/em_gmm.csv"
    # We load the data
    data = readcsv(infile)

    # configure a em_gmm init object
    algo1 = d4p.em_gmm_init(nComponents)
    # and compute initial model
    result1 = algo1.compute(data)

    # configure a em_gmm object
    algo2 = d4p.em_gmm(nComponents)

    # and compute em_gmm using initial weights and means
    result2 = algo2.compute(data, result1.weights, result1.means, result1.covariances)

    # implicit als prediction result objects provide covariances, goalFunction, means, nIterations and weights
    return result2


if __name__ == "__main__":
    res = main()
    print("Weights:\n", res.weights)
    print("Means:\n", res.means)
    for c in res.covariances:
        print("Covariance:\n", c)
    print('All looks good!')
