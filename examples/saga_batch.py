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

# daal4py Saga example for shared memory systems

import daal4py as d4p
import numpy as np

# let's try to use pandas' fast csv reader
try:
    import pandas
    read_csv = lambda f, c, t=np.float64: pandas.read_csv(f, usecols=c, delimiter=',', header=None, dtype=t)
except:
    # fall back to numpy loadtxt
    read_csv = lambda f, c, t=np.float64: np.loadtxt(f, usecols=c, delimiter=',', ndmin=2)


def main(readcsv=read_csv, method='defaultDense'):
    infile   = "./data/batch/XM.csv"
    # Read the data, let's have 3 independent variables
    data     = readcsv(infile, range(1))
    dep_data = readcsv(infile, range(1,2))
    nVectors = data.shape[0]

    # configure a Logistic Loss object
    logloss_algo = d4p.optimization_solver_logistic_loss(numberOfTerms=nVectors,
                                                         penaltyL1=0.3,
                                                         penaltyL2=0,
                                                         interceptFlag=True,
                                                         resultsToCompute='gradient')
    logloss_algo.setup(data, dep_data)

    # configure an Saga object
    lr = np.array([[0.01]], dtype=np.double)
    niters=100000
    saga_algo = d4p.optimization_solver_saga(nIterations=niters,
                                             accuracyThreshold=1e-5,
                                             batchSize=1,
                                             function=logloss_algo,
                                             learningRateSequence=lr,
                                             optionalResultRequired=True)


    # finally do the computation
    inp = np.zeros((2, 1), dtype=np.double)
    res = saga_algo.compute(inp, None)

    # The Saga result provides minimum and nIterations
    assert res.minimum.shape == inp.shape and res.nIterations[0][0] <= niters
    assert np.allclose(res.minimum, [[-0.17663868],[ 0.35893627]])

    return res


if __name__ == "__main__":
    res = main()
    print("\nMinimum:\n", res.minimum)
    print("\nNumber of iterations performed:\n", res.nIterations[0][0])
    print('All looks good!')
