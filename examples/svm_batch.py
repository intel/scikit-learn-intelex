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

# daal4py SVM example for shared memory systems

import daal4py as d4p
import numpy as np

# let's try to use pandas' fast csv reader
try:
    import pandas
    read_csv = lambda f, c, t=np.float64: pandas.read_csv(f, usecols=c, delimiter=',', header=None, dtype=t)
except:
    # fall back to numpy loadtxt
    read_csv = lambda f, c, t=np.float64: np.loadtxt(f, usecols=c, delimiter=',', ndmin=2)


def main(readcsv=read_csv, method='defaultDense'):
    # input data file
    infile = "./data/batch/svm_two_class_train_dense.csv"
    testfile = "./data/batch/svm_two_class_test_dense.csv"

    # Configure a SVM object to use rbf kernel (and adjusting cachesize)
    kern = d4p.kernel_function_linear(method=method)  # need an object that lives when creating train_algo
    train_algo = d4p.svm_training(doShrinking=True, kernel=kern, cacheSize=600000000)
    
    # Read data. Let's use features per observation
    data   = readcsv(infile, range(20))
    labels = readcsv(infile, range(20,21))
    train_result = train_algo.compute(data, labels)

    # Now let's do some prediction
    predict_algo = d4p.svm_prediction(kernel=kern)
    # read test data (with same #features)
    pdata = readcsv(testfile, range(20))
    plabels = readcsv(testfile, range(20,21))
    # now predict using the model from the training above
    predict_result = predict_algo.compute(pdata, train_result.model)

    # Prediction result provides prediction
    assert(predict_result.prediction.shape == (pdata.shape[0], 1))

    return (predict_result, plabels)


if __name__ == "__main__":
    (predict_result, plabels) = main()
    print("\nSVM classification results (first 20 observations):\n", predict_result.prediction[0:20])
    print("\nGround truth (first 20 observations):\n", plabels[0:20])
    print('All looks good!')
