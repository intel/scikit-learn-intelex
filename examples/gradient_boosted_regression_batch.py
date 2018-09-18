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

# daal4py Gradient Bossting Regression example for shared memory systems

import daal4py as d4p
from numpy import loadtxt, allclose

if __name__ == "__main__":
    maxIterations = 40

    # input data file
    infile = "./data/batch/df_regression_train.csv"

    # Configure a training object
    train_algo = d4p.gbt_regression_training(maxIterations=maxIterations)
    
    # Read data. Let's use 3 features per observation
    data   = loadtxt(infile, delimiter=',', usecols=range(13))
    deps = loadtxt(infile, delimiter=',', usecols=range(13,14))
    deps.shape = (deps.size, 1) # must be a 2d array
    train_result = train_algo.compute(data, deps)

    # Now let's do some prediction
    predict_algo = d4p.gbt_regression_prediction()
    # read test data (with same #features)
    pdata = loadtxt("./data/batch/df_regression_test.csv", delimiter=',', usecols=range(13))
    ptdata = loadtxt("./data/batch/df_regression_test.csv", delimiter=',', usecols=range(13,14))
    ptdata.shape = (ptdata.size, 1)
    # now predict using the model from the training above
    predict_result = predict_algo.compute(pdata, train_result.model)

    # Prediction result provides prediction
    assert(predict_result.prediction.shape == (pdata.shape[0], 1))

    print("\nGradient boosted trees prediction results (first 10 rows):\n", predict_result.prediction[0:10])
    print("\nGround truth (first 10 rows):\n", ptdata[0:10])
    print('All looks good!')
