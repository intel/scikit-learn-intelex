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

# daal4py SVM example for shared memory systems

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
    # input data file
    infile = "./data/batch/svm_two_class_train_dense.csv"
    testfile = "./data/batch/svm_two_class_test_dense.csv"

    # Configure a SVM object to use rbf kernel (and adjusting cachesize)
    kern = d4p.kernel_function_linear()
    # need an object that lives when creating train_algo
    train_algo = d4p.svm_training(method='thunder', kernel=kern, cacheSize=600000000)

    # Read data. Let's use features per observation
    data = readcsv(infile, range(20))
    labels = readcsv(infile, range(20, 21))
    train_result = train_algo.compute(data, labels)

    # Now let's do some prediction
    predict_algo = d4p.svm_prediction(kernel=kern)
    # read test data (with same #features)
    pdata = readcsv(testfile, range(20))
    plabels = readcsv(testfile, range(20, 21))
    # now predict using the model from the training above
    predict_result = predict_algo.compute(pdata, train_result.model)

    # Prediction result provides prediction
    assert(predict_result.prediction.shape == (pdata.shape[0], 1))

    # result of classification
    decision_result = predict_result.prediction
    predict_labels = np.where(decision_result >= 0, 1, -1)

    return (decision_result, predict_labels, plabels)


if __name__ == "__main__":
    (decision_function, predict_labels, plabels) = main()

    print(
        "\nSVM classification decision function (first 20 observations):\n",
        decision_function[0:20]
    )
    print("\nSVM classification results (first 20 observations):\n", predict_labels[0:20])
    print("\nGround truth (first 20 observations):\n", plabels[0:20])
    print('All looks good!')
