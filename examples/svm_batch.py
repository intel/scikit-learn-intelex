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

# daal4py SVM example for shared memory systems

import daal4py as d4p
from numpy import loadtxt, allclose

if __name__ == "__main__":

    # input data file
    infile = "./data/batch/svm_two_class_train_dense.csv"

    # Configure a SVM object to use rbf kernel (and adjusting cachesize)
    kern = d4p.kernel_function_rbf()  # need an object that lives when creating train_algo
    train_algo = d4p.svm_training(kernel=kern, cacheSize=600000000)
    
    # Read data. Let's use features per observation
    data   = loadtxt(infile, delimiter=',', usecols=range(20))
    labels = loadtxt(infile, delimiter=',', usecols=range(20,21))
    labels.shape = (labels.size, 1) # must be a 2d array
    train_result = train_algo.compute(data, labels)

    # Now let's do some prediction
    predict_algo = d4p.svm_prediction()
    # read test data (with same #features)
    pdata = loadtxt("./data/batch/svm_two_class_test_dense.csv", delimiter=',', usecols=range(20))
    # now predict using the model from the training above
    predict_result = predict_algo.compute(pdata, train_result.model)

    # Prediction result provides prediction
    assert(predict_result.prediction.shape == (data.shape[0], 1))

    print(predict_result.prediction)
    print('All looks good!')
