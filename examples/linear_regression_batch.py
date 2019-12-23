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
from daal4py.oneapi import sycl_context, sycl_buffer

# let's try to use pandas' fast csv reader
try:
    import pandas
    read_csv = lambda f, c, t=np.float64: pandas.read_csv(f, usecols=c, delimiter=',', header=None, dtype=t)
except:
    # fall back to numpy loadtxt
    read_csv = lambda f, c, t=np.float64: np.loadtxt(f, usecols=c, delimiter=',', ndmin=2)


# Commone code for both CPU and GPU computations
def compute(train_indep_data, train_dep_data, test_indep_data):
    # Configure a Linear regression training object
    train_algo = d4p.linear_regression_training(interceptFlag=True)
    # Now train/compute, the result provides the model for prediction
    train_result = train_algo.compute(train_indep_data, train_dep_data)
    # Now let's do some prediction
    predict_algo = d4p.linear_regression_prediction()
    # now predict using the model from the training above
    return predict_algo.compute(test_indep_data, train_result.model), train_result


# At this moment with sycl we are working only with numpy arrays
def to_numpy(data):
    try:
        from pandas import DataFrame
        if isinstance(data, DataFrame):
            return np.ascontiguousarray(data.values)
    except:
        pass
    try:
        from scipy.sparse import csr_matrix
        if isinstance(data, csr_matrix):
            return data.toarray()
    except:
        pass
    return data


def main(readcsv=read_csv, method='defaultDense'):
    # read training data. Let's have 10 independent, and 2 dependent variables (for each observation)
    trainfile = "./data/batch/linear_regression_train.csv"
    train_indep_data = readcsv(trainfile, range(10))
    train_dep_data = readcsv(trainfile, range(10,12))

    # read testing data
    testfile = "./data/batch/linear_regression_test.csv"
    test_indep_data = readcsv(testfile, range(10))
    test_dep_data = readcsv(testfile, range(10,12))

    # Using of the classic way (computations on CPU)
    result_classic, train_result = compute(train_indep_data, train_dep_data, test_indep_data)

    train_indep_data = to_numpy(train_indep_data)
    train_dep_data = to_numpy(train_dep_data)
    test_indep_data = to_numpy(test_indep_data)

    # It is possible to specify to make the computations on GPU
    with sycl_context('gpu'):
        sycl_train_indep_data = sycl_buffer(train_indep_data)
        sycl_train_dep_data = sycl_buffer(train_dep_data)
        sycl_test_indep_data = sycl_buffer(test_indep_data)
        result_gpu, _ = compute(sycl_train_indep_data, sycl_train_dep_data, sycl_test_indep_data)

    # The prediction result provides prediction
    assert result_classic.prediction.shape == (test_dep_data.shape[0], test_dep_data.shape[1])

    assert np.allclose(result_classic.prediction, result_gpu.prediction)

    return (train_result, result_classic, test_dep_data)


if __name__ == "__main__":
    (train_result, predict_result, test_dep_data) = main()
    print("\nLinear Regression coefficients:\n", train_result.model.Beta)
    print("\nLinear Regression prediction results: (first 10 rows):\n", predict_result.prediction[0:10])
    print("\nGround truth (first 10 rows):\n", test_dep_data[0:10])
    print('All looks good!')
