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

# daal4py SGD (Stochastic Gradient Descent) example for shared memory systems
# using Logisitc Loss objective function

import daal4py as d4p
import numpy as np

# let's try to use pandas' fast csv reader
try:
    import pandas
    read_csv = lambda f, c: pandas.read_csv(f, usecols=c, delimiter=',', header=None).values
except:
    # fall back to numpy loadtxt
    read_csv = lambda f, c: np.loadtxt(f, usecols=c, delimiter=',')


def main():
    infile   = "./data/batch/custom.csv"
    # Read the data, let's have 4 independent variables
    data     = read_csv(infile, range(4))
    dep_data = read_csv(infile, range(4,5))
    nVectors = data.shape[0]
    dep_data.shape = (nVectors, 1) # must be a 2d array

    # configure a logistic loss object
    ll_algo = d4p.optimization_solver_logistic_loss(nVectors)
    ll_algo.setup(data, dep_data, None)

    # configure a SGD object
    lrs = np.array([[0.01]], dtype=np.double)
    niters = 1000
    sgd_algo = d4p.optimization_solver_sgd(ll_algo,
                                           learningRateSequence=lrs,
                                           accuracyThreshold=0.02,
                                           nIterations=niters)

    # finally do the computation
    inp = np.array([[1], [1], [1], [1], [1]], dtype=np.double)
    res = sgd_algo.compute(inp)

    # The SGD result provides minimum and nIterations
    assert res.minimum.shape == inp.shape and res.nIterations[0][0] <= niters

    return res


if __name__ == "__main__":
    res = main()
    print("\nMinimum:\n", res.minimum)
    print("\nNumber of iterations performed:\n", res.nIterations[0][0])
    print('All looks good!')
