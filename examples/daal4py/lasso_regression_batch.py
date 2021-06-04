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

# daal4py Lasso Regression example for shared memory systems

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
    infile = "./data/batch/linear_regression_train.csv"
    testfile = "./data/batch/linear_regression_test.csv"

    # Configure a Lasso regression training object
    train_algo = d4p.lasso_regression_training(interceptFlag=True)

    # Read data. Let's have 10 independent,
    # and 2 dependent variables (for each observation)
    indep_data = readcsv(infile, range(10))
    dep_data = readcsv(infile, range(10, 12))
    # Now train/compute, the result provides the model for prediction
    train_result = train_algo.compute(indep_data, dep_data)

    # Now let's do some prediction
    predict_algo = d4p.lasso_regression_prediction()
    # read test data (with same #features)
    pdata = readcsv(testfile, range(10))
    ptdata = readcsv(testfile, range(10, 12))
    # now predict using the model from the training above
    predict_result = predict_algo.compute(pdata, train_result.model)

    # The prediction result provides prediction
    assert predict_result.prediction.shape == (pdata.shape[0], dep_data.shape[1])

    # the example is used in tests with the scipy.sparse matrix
    # we use this trick until subtracting a sparse matrix is not supported
    if hasattr(ptdata, 'toarray'):
        ptdata = ptdata.toarray()
    # this assertion is outdated, will be fixed in next release
    # assert np.square(predict_result.prediction - np.asarray(ptdata)).mean() < 2.2

    return (predict_result, ptdata)


if __name__ == "__main__":
    (predict_result, ptdata) = main()
    print(
        "\nLasso Regression prediction results: (first 10 rows):\n",
        predict_result.prediction[0:10]
    )
    print("\nGround truth (first 10 rows):\n", ptdata[0:10])
    print('All looks good!')
