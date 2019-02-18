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

# daal4py Linear Regression example for shared memory systems

import daal4py as d4p
import numpy as np
from hpat import jit

# let's try to use pandas' fast csv reader
try:
    import pandas
    read_csv = lambda f, c, t=np.float64: pandas.read_csv(f, usecols=c, delimiter=',', header=None, dtype=t)
except:
    # fall back to numpy loadtxt
    read_csv = lambda f, c, t=np.float64: np.loadtxt(f, usecols=c, delimiter=',', ndmin=2)


@jit(nopython=True)
def main(method='defaultDense'):
    infile = "./data/batch/linear_regression_train.csv"
    testfile = "./data/batch/linear_regression_test.csv"

    # Configure a Linear regression training object
    train_algo = d4p.linear_regression_training(interceptFlag=True)
    
    # Read data. Let's have 10 independent, and 2 dependent variables (for each observation)
    indep_data = pandas.read_csv(infile, names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], usecols=[0,1,2,3,4,5,6,7,8,9], delimiter=',', header=None, dtype={'0':np.float64, '1':np.float64, '2':np.float64, '3':np.float64, '4':np.float64, '5':np.float64, '6':np.float64, '7':np.float64, '8':np.float64, '9':np.float64})
    dep_data   = pandas.read_csv(infile, names=['10', '11'], usecols=[10,11], delimiter=',', header=None, dtype={'10':np.float64, '11':np.float64})
    # Now train/compute, the result provides the model for prediction
    train_result = train_algo.compute(indep_data[:], dep_data[:])

    # Now let's do some prediction
    predict_algo = d4p.linear_regression_prediction()
    # read test data (with same #features)
    pdata = pandas.read_csv(testfile, names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], usecols=[0,1,2,3,4,5,6,7,8,9], delimiter=',', header=None, dtype={'0':np.float64, '1':np.float64, '2':np.float64, '3':np.float64, '4':np.float64, '5':np.float64, '6':np.float64, '7':np.float64, '8':np.float64, '9':np.float64})
    ptdata = pandas.read_csv(testfile, names=['10', '11'], usecols=[10,11], delimiter=',', header=None, dtype={'10':np.float64, '11':np.float64})
    # now predict using the model from the training above
    predict_result = predict_algo.compute(pdata, train_result.model)

    # The prediction result provides prediction
    assert predict_result.prediction.shape == (pdata.shape[0], dep_data.shape[1])

    return (train_result, predict_result, ptdata)


if __name__ == "__main__":
    (train_result, predict_result, ptdata) = main()
    print("\nLinear Regression coefficients:\n", train_result.model.Beta)
    print("\nLinear Regression prediction results: (first 10 rows):\n", predict_result.prediction[0:10])
    print("\nGround truth (first 10 rows):\n", ptdata[0:10])
    print('All looks good!')
