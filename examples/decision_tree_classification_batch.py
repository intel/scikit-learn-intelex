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

# daal4py Decision Tree Classification example for shared memory systems

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
    # input data file
    infile = "./data/batch/decision_tree_train.csv"
    prunefile = "./data/batch/decision_tree_prune.csv"
    testfile = "./data/batch/decision_tree_test.csv"

    # Configure a training object (5 classes)
    train_algo = d4p.decision_tree_classification_training(5)

    # Read data. Let's use 5 features per observation
    data = readcsv(infile, range(5), t=np.float32)
    labels = readcsv(infile, range(5, 6), t=np.float32)
    prunedata = readcsv(prunefile, range(5), t=np.float32)
    prunelabels = readcsv(prunefile, range(5, 6), t=np.float32)
    train_result = train_algo.compute(data, labels, prunedata, prunelabels)

    # Now let's do some prediction
    predict_algo = d4p.decision_tree_classification_prediction()
    # read test data (with same #features)
    pdata = readcsv(testfile, range(5), t=np.float32)
    plabels = readcsv(testfile, range(5, 6), t=np.float32)
    # now predict using the model from the training above
    predict_result = predict_algo.compute(pdata, train_result.model)

    # Prediction result provides prediction
    assert(predict_result.prediction.shape == (pdata.shape[0], 1))

    return (train_result, predict_result, plabels)


if __name__ == "__main__":
    (train_result, predict_result, plabels) = main()
    print(
        "\nDecision tree prediction results (first 20 rows):\n",
        predict_result.prediction[0:20]
    )
    print("\nGround truth (first 20 rows):\n", plabels[0:20])
    print('All looks good!')
