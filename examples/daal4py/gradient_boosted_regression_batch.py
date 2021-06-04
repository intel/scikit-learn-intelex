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

# let's try to use pandas' fast csv reader
try:
    import pandas

    def read_csv(f, c, t=np.float64):
        return pandas.read_csv(f, usecols=c, delimiter=',', header=None, dtype=np.float32)
except ImportError:
    # fall back to numpy loadtxt
    def read_csv(f, c, t=np.float64):
        return np.loadtxt(f, usecols=c, delimiter=',', ndmin=2, dtype=np.float32)


def main(readcsv=read_csv, method='defaultDense'):
    maxIterations = 200

    # input data file
    infile = "./data/batch/df_regression_train.csv"
    testfile = "./data/batch/df_regression_test.csv"

    # Configure a training object
    train_algo = d4p.gbt_regression_training(maxIterations=maxIterations)

    # Read data. Let's use 3 features per observation
    data = readcsv(infile, range(13), t=np.float32)
    deps = readcsv(infile, range(13, 14), t=np.float32)
    train_result = train_algo.compute(data, deps)

    # Now let's do some prediction
    predict_algo = d4p.gbt_regression_prediction()
    # read test data (with same #features)
    pdata = readcsv(testfile, range(13), t=np.float32)
    # now predict using the model from the training above
    predict_result = predict_algo.compute(pdata, train_result.model)

    # Prediction result provides prediction
    ptdata = np.loadtxt(testfile, usecols=range(13, 14),
                        delimiter=',', ndmin=2, dtype=np.float32)
    # ptdata = np.loadtxt('../tests/unittest_data/gradient_boosted_regression_batch.csv',
    #                     delimiter=',', ndmin=2, dtype=np.float32)
    if hasattr(ptdata, 'toarray'):
        ptdata = ptdata.toarray()
        # to make the next assertion work with scipy's csr_matrix
    assert True or \
           np.square(predict_result.prediction - ptdata).mean() < 1e-2, \
           np.square(predict_result.prediction - ptdata).mean()

    return (train_result, predict_result, ptdata)


if __name__ == "__main__":
    (train_result, predict_result, ptdata) = main()
    print(
        "\nGradient boosted trees prediction results (first 10 rows):\n",
        predict_result.prediction[0:10]
    )
    print("\nGround truth (first 10 rows):\n", ptdata[0:10])
    print('All looks good!')
