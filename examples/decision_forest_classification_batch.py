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

# daal4py Decision Forest Classification example for shared memory systems

import daal4py as d4p
import numpy as np

# let's try to use pandas' fast csv reader
try:
    import pandas
    read_csv = lambda f, c, t=np.float64: pandas.read_csv(f, usecols=c, delimiter=',', header=None, dtype=t)
except:
    # fall back to numpy loadtxt
    read_csv = lambda f, c, t=np.float64: np.loadtxt(f, usecols=c, delimiter=',', ndmin=2, dtype=t)


def main(readcsv=read_csv, method='defaultDense'):
    # input data file
    infile = "./data/batch/df_classification_train.csv"
    testfile = "./data/batch/df_classification_test.csv"

    # Configure a training object (5 classes)
    train_algo = d4p.decision_forest_classification_training(5,
                                                             nTrees=10,
                                                             minObservationsInLeafNode=8,
                                                             featuresPerNode=3,
                                                             engine = d4p.engines_mt19937(seed=777),
                                                             varImportance='MDI',
                                                             bootstrap=True,
                                                             resultsToCompute='computeOutOfBagError')
    
    # Read data. Let's use 3 features per observation
    data   = readcsv(infile, range(3), t=np.float32)
    labels = readcsv(infile, range(3,4), t=np.float32)
    train_result = train_algo.compute(data, labels)
    # Traiing result provides (depending on parameters) model, outOfBagError, outOfBagErrorPerObservation and/or variableImportance

    # Now let's do some prediction
    predict_algo = d4p.decision_forest_classification_prediction(5)
    # read test data (with same #features)
    pdata = readcsv(testfile, range(3), t=np.float32)
    plabels = readcsv(testfile, range(3,4), t=np.float32)
    # now predict using the model from the training above
    predict_result = predict_algo.compute(pdata, train_result.model)

    # Prediction result provides prediction
    assert(predict_result.prediction.shape == (pdata.shape[0], 1))

    return (train_result, predict_result, plabels)


if __name__ == "__main__":
    (train_result, predict_result, plabels) = main()
    print("\nVariable importance results:\n", train_result.variableImportance)
    print("\nOOB error:\n", train_result.outOfBagError)
    print("\nDecision forest prediction results (first 10 rows):\n", predict_result.prediction[0:10])
    print("\nGround truth (first 10 rows):\n", plabels[0:10])
    print('All looks good!')
