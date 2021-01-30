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

# daal4py Gradient Bossting Regression example for shared memory systems

import daal4py as d4p
import numpy as np
import os
from daal4py.oneapi import sycl_buffer

# let's try to use pandas' fast csv reader
try:
    import pandas

    def read_csv(f, c, t=np.float64):
        return pandas.read_csv(f, usecols=c, delimiter=',', header=None, dtype=np.float32)
except ImportError:
    # fall back to numpy loadtxt
    def read_csv(f, c, t=np.float64):
        return np.loadtxt(f, usecols=c, delimiter=',', ndmin=2, dtype=np.float32)


# Commone code for both CPU and GPU computations
def compute(train_indep_data, train_dep_data, test_indep_data, maxIterations):
    # Configure a training object
    train_algo = d4p.gbt_regression_training(maxIterations=maxIterations, fptype='float')
    train_result = train_algo.compute(train_indep_data, train_dep_data)
    # Now let's do some prediction
    predict_algo = d4p.gbt_regression_prediction(fptype='float')
    # now predict using the model from the training above
    return predict_algo.compute(test_indep_data, train_result.model)


# At this moment with sycl we are working only with numpy arrays
def to_numpy(data):
    try:
        from pandas import DataFrame
        if isinstance(data, DataFrame):
            return np.ascontiguousarray(data.values)
    except ImportError:
        pass
    try:
        from scipy.sparse import csr_matrix
        if isinstance(data, csr_matrix):
            return data.toarray()
    except ImportError:
        pass
    return data


def main(readcsv=read_csv, method='defaultDense'):
    maxIterations = 200

    # input data file
    infile = os.path.join('..', 'data', 'batch', 'df_regression_train.csv')
    testfile = os.path.join('..', 'data', 'batch', 'df_regression_test.csv')

    # Read data. Let's use 13 features per observation
    train_indep_data = readcsv(infile, range(13), t=np.float32)
    train_dep_data = readcsv(infile, range(13, 14), t=np.float32)
    # read test data (with same #features)
    test_indep_data = readcsv(testfile, range(13), t=np.float32)

    # Using of the classic way (computations on CPU)
    result_classic = compute(train_indep_data, train_dep_data,
                             test_indep_data, maxIterations)

    train_indep_data = to_numpy(train_indep_data)
    train_dep_data = to_numpy(train_dep_data)
    test_indep_data = to_numpy(test_indep_data)

    try:
        from dpctx import device_context, device_type

        def gpu_context():
            return device_context(device_type.gpu, 0)
    except:
        from daal4py.oneapi import sycl_context

        def gpu_context():
            return sycl_context('gpu')

    # It is possible to specify to make the computations on GPU
    with gpu_context():
        sycl_train_indep_data = sycl_buffer(train_indep_data)
        sycl_train_dep_data = sycl_buffer(train_dep_data)
        sycl_test_indep_data = sycl_buffer(test_indep_data)
        _ = compute(sycl_train_indep_data, sycl_train_dep_data,
                    sycl_test_indep_data, maxIterations)

    test_dep_data = np.loadtxt(testfile, usecols=range(13, 14), delimiter=',',
                               ndmin=2, dtype=np.float32)

    return (result_classic, test_dep_data)


if __name__ == "__main__":
    (predict_result, test_dep_data) = main()
    print(
        "\nGradient boosted trees prediction results (first 10 rows):\n",
        predict_result.prediction[0:10]
    )
    print("\nGround truth (first 10 rows):\n", test_dep_data[0:10])
    print('All looks good!')
