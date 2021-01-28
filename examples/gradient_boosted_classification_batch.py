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

# daal4py Gradient Bossting Classification example for shared memory systems

import daal4py as d4p
import numpy as np

# let's try to use pandas' fast csv reader
try:
    import pandas

    def read_csv(f, c=None, t=np.float64):
        return pandas.read_csv(f, usecols=c, delimiter=',', header=None, dtype=t)
except ImportError:
    # fall back to numpy loadtxt
    def read_csv(f, c=None, t=np.float64):
        return np.loadtxt(f, usecols=c, delimiter=',', ndmin=2, dtype=t)


def main(readcsv=read_csv, method='defaultDense'):
    nFeatures = 3
    nClasses = 5
    maxIterations = 200
    minObservationsInLeafNode = 8
    # input data file
    infile = "./data/batch/df_classification_train.csv"
    testfile = "./data/batch/df_classification_test.csv"

    # Configure a training object (5 classes)
    train_algo = d4p.gbt_classification_training(
        nClasses=nClasses,
        maxIterations=maxIterations,
        minObservationsInLeafNode=minObservationsInLeafNode,
        featuresPerNode=nFeatures,
        varImportance='weight|totalCover|cover|totalGain|gain'
    )

    # Read data. Let's use 3 features per observation
    data = readcsv(infile, range(3), t=np.float32)
    labels = readcsv(infile, range(3, 4), t=np.float32)
    train_result = train_algo.compute(data, labels)

    # Now let's do some prediction
    # previous version has different interface
    predict_algo = d4p.gbt_classification_prediction(
        nClasses=nClasses,
        resultsToEvaluate="computeClassLabels|computeClassProbabilities"
    )
    # read test data (with same #features)
    pdata = readcsv(testfile, range(3), t=np.float32)
    # now predict using the model from the training above
    predict_result = predict_algo.compute(pdata, train_result.model)

    # Prediction result provides prediction
    plabels = readcsv(testfile, range(3, 4), t=np.float32)
    assert np.count_nonzero(predict_result.prediction - plabels) / pdata.shape[0] < 0.022

    return (train_result, predict_result, plabels)


if __name__ == "__main__":
    (train_result, predict_result, plabels) = main()
    print(
        "\nGradient boosted trees prediction results (first 10 rows):\n",
        predict_result.prediction[0:10]
    )
    print("\nGround truth (first 10 rows):\n", plabels[0:10])
    print(
        "\nGradient boosted trees prediction probabilities (first 10 rows):\n",
        predict_result.probabilities[0:10]
    )
    print("\nvariableImportanceByWeight:\n", train_result.variableImportanceByWeight)
    print(
        "\nvariableImportanceByTotalCover:\n",
        train_result.variableImportanceByTotalCover
    )
    print("\nvariableImportanceByCover:\n", train_result.variableImportanceByCover)
    print(
        "\nvariableImportanceByTotalGain:\n",
        train_result.variableImportanceByTotalGain
    )
    print("\nvariableImportanceByGain:\n", train_result.variableImportanceByGain)
    print('All looks good!')
