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

# daal4py Naive Bayes Classification example for streaming on shared memory systems

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
    # input data file
    infile = "./data/batch/naivebayes_train_dense.csv"
    testfile = "./data/batch/naivebayes_test_dense.csv"

    # Configure a training object (20 classes)
    train_algo = d4p.multinomial_naive_bayes_training(20, streaming=True, method=method)

    chunk_size = 250
    lines_read = 0
    # read and feed chunk by chunk
    while True:
        # Read data in chunks
        # Read data. Let's use 20 features per observation
        try:
            data = readcsv(infile, range(20), lines_read, chunk_size)
            labels = readcsv(infile, range(20, 21), lines_read, chunk_size)
        except:
            break
        # Now feed chunk
        train_algo.compute(data, labels)
        lines_read += data.shape[0]

    # All chunks are done, now finalize the computation
    train_result = train_algo.finalize()

    # Now let's do some prediction
    pred_algo = d4p.multinomial_naive_bayes_prediction(20, method=method)
    # read test data (with same #features)
    pred_data = readcsv(testfile, range(20))
    pred_labels = readcsv(testfile, range(20, 21))
    # now predict using the model from the training above
    pred_result = pred_algo.compute(pred_data, train_result.model)

    # Prediction result provides prediction
    assert(pred_result.prediction.shape == (pred_data.shape[0], 1))

    return (pred_result, pred_labels)


if __name__ == "__main__":
    (result, labels) = main()
    print(
        "\nNaiveBayes classification results (first 20 observations):\n",
        result.prediction[0:20]
    )
    print("\nGround truth (first 20 observations)\n", labels[0:20])
    print('All looks good!')
