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

# daal4py LBFGS (limited memory Broyden-Fletcher-Goldfarb-Shanno) example for shared memory systems
# using Mean Squared Error objective function

import daal4py as d4p
import numpy as np

# let's try to use pandas' fast csv reader
try:
    import pandas
    read_csv = lambda f,c: pandas.read_csv(f, usecols=c, delimiter=',').values
except:
    # fall back to numpy loadtxt
    read_csv = lambda f,c: np.loadtxt(f, usecols=c, delimiter=',')


def main():
    infile   = "./data/batch/lbfgs.csv"
    # Read the data, let's have 10 independent variables
    data     = read_csv(infile, range(10))
    dep_data = read_csv(infile, range(10,11))
    nVectors = data.shape[0]
    dep_data.shape = (nVectors, 1) # must be a 2d array

    # configure a MSE object
    mse_algo = d4p.optimization_solver_mse(nVectors)
    mse_algo.setup(data, dep_data, None)

    # configure an LBFGS object
    sls = np.array([[1.0e-4]], dtype=np.double)
    niters = 1000
    lbfgs_algo = d4p.optimization_solver_lbfgs(mse_algo,
                                               stepLengthSequence=sls,
                                               nIterations=niters)

    # finally do the computation
    inp = np.array([[100]]*11, dtype=np.double)
    res = lbfgs_algo.compute(inp)

    # The LBFGS result provides minimum and nIterations
    assert res.minimum.shape == inp.shape and res.nIterations[0][0] <= niters


if __name__ == "__main__":
    main()
    print('All looks good!')
