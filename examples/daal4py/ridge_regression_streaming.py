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

# daal4py Ridge Regression example for streaming on shared memory systems

import daal4py as d4p
import numpy as np

# let's try to use pandas' fast csv reader
try:
    import pandas

    def read_csv(f, c, s=0, n=None, t=np.float64):
        return pandas.read_csv(f, usecols=c, delimiter=',', header=None,
                               skiprows=s, nrows=n, dtype=t)
except:
    # fall back to numpy genfromtxt
    def read_csv(f, c, s=0, n=np.iinfo(np.int64).max):
        a = np.genfromtxt(f, usecols=c, delimiter=',', skip_header=s, max_rows=n)
        if a.shape[0] == 0:
            raise Exception("done")
        if a.ndim == 1:
            return a[:, np.newaxis]
        return a


def main(readcsv=read_csv, method='defaultDense'):
    infile = "./data/batch/linear_regression_train.csv"
    testfile = "./data/batch/linear_regression_test.csv"

    # Configure a Ridge regression training object for streaming
    train_algo = d4p.ridge_regression_training(interceptFlag=True, streaming=True)

    chunk_size = 250
    lines_read = 0
    # read and feed chunk by chunk
    while True:
        # Read data in chunks
        # Let's have 10 independent, and 2 dependent variables (for each observation)
        try:
            indep_data = readcsv(infile, range(10), lines_read, chunk_size)
            dep_data = readcsv(infile, range(10, 12), lines_read, chunk_size)
        except:
            break
        # Now feed chunk
        train_algo.compute(indep_data, dep_data)
        lines_read += indep_data.shape[0]

    # All chunks are done, now finalize the computation
    train_result = train_algo.finalize()

    # Now let's do some prediction
    predict_algo = d4p.ridge_regression_prediction()
    # read test data (with same #features)
    pdata = readcsv(testfile, range(10))
    ptdata = readcsv(testfile, range(10, 12))
    # now predict using the model from the training above
    predict_result = predict_algo.compute(pdata, train_result.model)

    # The prediction result provides prediction
    assert predict_result.prediction.shape == (pdata.shape[0], dep_data.shape[1])

    return (predict_result, ptdata)


if __name__ == "__main__":
    (predict_result, ptdata) = main()
    print(
        "\nRidge Regression prediction results: (first 10 rows):\n",
        predict_result.prediction[0:10]
    )
    print("\nGround truth (first 10 rows):\n", ptdata[0:10])
    print('All looks good!')
