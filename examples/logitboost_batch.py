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

# daal4py Logitboost example for shared memory systems

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
    infile = "./data/batch/logitboost_train.csv"
    testfile = "./data/batch/logitboost_test.csv"
    nClasses = 5


    # Configure a logitboost training object
    train_algo = d4p.logitboost_training(nClasses, maxIterations=100, accuracyThreshold=0.01)
    
    # Read data. Let's have 20 independent, and 1 dependent variable (for each observation)
    indep_data = readcsv(infile, range(20))
    dep_data   = readcsv(infile, range(20,21))
    # Now train/compute, the result provides the model for prediction
    train_result = train_algo.compute(indep_data, dep_data)

    # Now let's do some prediction
    predict_algo = d4p.logitboost_prediction(nClasses)
    # read test data (with same #features)
    pdata = readcsv(testfile, range(20))
    # now predict using the model from the training above
    predict_result = predict_algo.compute(pdata, train_result.model)

    # The prediction result provides prediction
    assert predict_result.prediction.shape == (pdata.shape[0], dep_data.shape[1])
    ptdata = np.loadtxt(testfile, usecols=range(20,21), delimiter=',', ndmin=2)
    assert np.allclose(predict_result.prediction, ptdata)
    return (train_result, predict_result, ptdata)


if __name__ == "__main__":
    (train_result, predict_result, ptdata) = main()
    print("\nGround truth (first 20 observations):\n", ptdata[:20])
    print("Logitboost classification results: (first 20 observations):\n", predict_result.prediction[:20])
    print('All looks good!')
