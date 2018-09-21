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

# daal4py Gradient Bossting Classification example for shared memory systems

import daal4py as d4p
import numpy as np

# let's try to use pandas' fast csv reader
try:
    import pandas
    read_csv = lambda f, c: pandas.read_csv(f, usecols=c, delimiter=',', header=None).values
except:
    # fall back to numpy loadtxt
    read_csv = lambda f, c: np.loadtxt(f, usecols=c, delimiter=',')


def main():
    nFeatures = 3
    nClasses = 5
    maxIterations = 40
    minObservationsInLeafNode = 8
    # input data file
    infile = "./data/batch/df_classification_train.csv"
    testfile = "./data/batch/df_classification_test.csv"

    # Configure a training object (5 classes)
    train_algo = d4p.gbt_classification_training(nClasses=nClasses, maxIterations=maxIterations, minObservationsInLeafNode=minObservationsInLeafNode, featuresPerNode=nFeatures)

    # Read data. Let's use 3 features per observation
    data   = read_csv(infile, range(3))
    labels = read_csv(infile, range(3,4))
    labels.shape = (labels.size, 1) # must be a 2d array
    train_result = train_algo.compute(data, labels)

    # Now let's do some prediction
    predict_algo = d4p.gbt_classification_prediction(5)
    # read test data (with same #features)
    pdata = read_csv(testfile, range(3))
    plabels = read_csv(testfile, range(3,4))
    plabels.shape = (plabels.size, 1) # must be a 2d array
    # now predict using the model from the training above
    predict_result = predict_algo.compute(pdata, train_result.model)

    # Prediction result provides prediction
    assert(predict_result.prediction.shape == (pdata.shape[0], 1))

    return (train_result, predict_result, plabels)


if __name__ == "__main__":
    (train_result, predict_result, plabels) = main()
    print("\nGradient boosted trees prediction results (first 10 rows):\n", predict_result.prediction[0:10])
    print("\nGround truth (first 10 rows):\n", plabels[0:10])
    print('All looks good!')
