#*******************************************************************************
# Copyright 2014-2018 Intel Corporation
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
from numpy import loadtxt, allclose

if __name__ == "__main__":

    # input data file
    infile = "./data/batch/df_classification_train.csv"

    # Configure a training object (5 classes)
    train_algo = d4p.decision_forest_classification_training(5, nTrees=10, minObservationsInLeafNode=8, featuresPerNode=3,
                                                             varImportance='MDI', bootstrap=True, resultsToCompute='computeOutOfBagError')
    
    # Read data. Let's use 3 features per observation
    data   = loadtxt(infile, delimiter=',', usecols=range(3))
    labels = loadtxt(infile, delimiter=',', usecols=range(3,4))
    labels.shape = (labels.size, 1) # must be a 2d array
    train_result = train_algo.compute(data, labels)
    # Traiing result provides (depending on parameters) model, outOfBagError, outOfBagErrorPerObservation and/or variableImportance

    # Now let's do some prediction
    predict_algo = d4p.decision_forest_classification_prediction(5)
    # read test data (with same #features)
    pdata = loadtxt("./data/batch/df_classification_test.csv", delimiter=',', usecols=range(3))
    plabels = loadtxt("./data/batch/df_classification_test.csv", delimiter=',', usecols=range(3,4))
    plabels.shape = (plabels.size, 1)
    # now predict using the model from the training above
    predict_result = predict_algo.compute(pdata, train_result.model)

    # Prediction result provides prediction
    assert(predict_result.prediction.shape == (pdata.shape[0], 1))

    print("\nVariable importance results:\n", train_result.variableImportance)
    print("\nOOB error:\n", train_result.outOfBagError)
    print("\nDecision forest prediction results (first 10 rows):\n", predict_result.prediction[0:10])
    print("\nGround truth (first 10 rows):\n", plabels[0:10])
    print('All looks good!')
