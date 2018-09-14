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

# daal4py Decision Tree Classification example for shared memory systems

import daal4py as d4p
from numpy import loadtxt, allclose

if __name__ == "__main__":

    # input data file
    infile = "./data/batch/decision_tree_train.csv"
    prunefile = "./data/batch/decision_tree_prune.csv"

    # Configure a training object (5 classes)
    train_algo = d4p.decision_tree_classification_training(5)
    
    # Read data. Let's use 5 features per observation
    data   = loadtxt(infile, delimiter=',', usecols=range(5))
    labels = loadtxt(infile, delimiter=',', usecols=range(5,6))
    prunedata = loadtxt(prunefile, delimiter=',', usecols=range(5))
    prunelabels = loadtxt(prunefile, delimiter=',', usecols=range(5,6))
    labels.shape = (labels.size, 1) # must be a 2d array
    prunelabels.shape = (prunelabels.size, 1) # must be a 2d array
    train_result = train_algo.compute(data, labels, prunedata, prunelabels)

    # Now let's do some prediction
    predict_algo = d4p.decision_tree_classification_prediction()
    # read test data (with same #features)
    pdata = loadtxt("./data/batch/decision_tree_test.csv", delimiter=',', usecols=range(5))
    plabels = loadtxt("./data/batch/decision_tree_test.csv", delimiter=',', usecols=range(5,6))
    plabels.shape = (plabels.size, 1)
    # now predict using the model from the training above
    predict_result = predict_algo.compute(pdata, train_result.model)

    # Prediction result provides prediction
    assert(predict_result.prediction.shape == (pdata.shape[0], 1))

    print("\nDecision tree prediction results (first 20 rows):\n", predict_result.prediction[0:20])
    print("\nGround truth (first 20 rows):\n", plabels[0:20])
    print('All looks good!')
