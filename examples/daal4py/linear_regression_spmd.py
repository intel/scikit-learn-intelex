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

# daal4py Linear Regression example for distributed memory systems; SPMD mode
# run like this:
#    mpirun -n 4 python ./linreg_spmd.py

import daal4py as d4p
from numpy import loadtxt

if __name__ == "__main__":
    # Initialize SPMD mode
    d4p.daalinit()

    # Each process gets its own data
    infile = "./data/distributed/linear_regression_train_" + \
        str(d4p.my_procid() + 1) + ".csv"

    # Configure a Linear regression training object
    train_algo = d4p.linear_regression_training(distributed=True)

    # Read data. Let's have 10 independent,
    # and 2 dependent variables (for each observation)
    indep_data = loadtxt(infile, delimiter=',', usecols=range(10))
    dep_data = loadtxt(infile, delimiter=',', usecols=range(10, 12))
    # Now train/compute, the result provides the model for prediction
    train_result = train_algo.compute(indep_data, dep_data)

    # Now let's do some prediction
    # It run only on a single node
    if d4p.my_procid() == 0:
        predict_algo = d4p.linear_regression_prediction()
        # read test data (with same #features)
        pdata = loadtxt("./data/distributed/linear_regression_test.csv",
                        delimiter=',', usecols=range(10))
        # now predict using the model from the training above
        predict_result = d4p.linear_regression_prediction().compute(pdata,
                                                                    train_result.model)

        # The prediction result provides prediction
        assert predict_result.prediction.shape == (pdata.shape[0], dep_data.shape[1])

    print('All looks good!')
    d4p.daalfini()
