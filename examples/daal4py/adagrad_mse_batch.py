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

# daal4py AdaGrad (Adaptive Subgradient Method) example for shared memory systems
# using Mean Squared Error objective function

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
    infile = "./data/batch/mse.csv"
    # Read the data, let's have 3 independent variables
    data = readcsv(infile, range(3))
    dep_data = readcsv(infile, range(3, 4))
    nVectors = data.shape[0]

    # configure a MSE object
    mse_algo = d4p.optimization_solver_mse(nVectors)
    mse_algo.setup(data, dep_data)

    # configure an AdaGrad object
    lr = np.array([[1.0]], dtype=np.double)
    niters = 1000
    sgd_algo = d4p.optimization_solver_adagrad(mse_algo,
                                               learningRate=lr,
                                               accuracyThreshold=0.0000001,
                                               nIterations=niters,
                                               batchSize=1)

    # finally do the computation
    inp = np.array([[8], [2], [1], [4]], dtype=np.double)
    res = sgd_algo.compute(inp)

    # The AdaGrad result provides minimum and nIterations
    assert res.minimum.shape == inp.shape and res.nIterations[0][0] <= niters

    return res


if __name__ == "__main__":
    res = main()
    print("\nMinimum:\n", res.minimum)
    print("\nNumber of iterations performed:\n", res.nIterations[0][0])
    print('All looks good!')
