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

# daal4py Linear Regression example for shared memory systems

import daal4py as d4p
from numpy import loadtxt, allclose

if __name__ == "__main__":

    infile = "./data/batch/linear_regression_train.csv"

    # Configure a Linear regression training object
    talgo = d4p.linear_regression_training()
    
    # Read data. Let's have 9 independent, and 2 dependent variables (for each observation)
    indep_data = loadtxt("./data/batch/linear_regression_train.csv", delimiter=',', usecols=range(9))
    dep_data   = loadtxt("./data/batch/linear_regression_train.csv", delimiter=',', usecols=range(9,11))
    # Now train/compute, the result provides the model for prediction
    tresult = talgo.compute(indep_data, dep_data)

    # Now let's do some prediction
    palgo = d4p.linear_regression_prediction()
    # read test data (with same #features)
    pdata = loadtxt("./data/batch/linear_regression_test.csv", delimiter=',', usecols=range(9))
    # now predict using the model from the training above
    presult = d4p.linear_regression_prediction().compute(pdata, tresult.model)

    # The prediction reulst provides prediction
    assert presult.prediction.shape == (pdata.shape[0], dep_data.shape[1])

    print('All looks good!')
