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

# daal4py BF KNN example for shared memory systems

import daal4py as d4p
import numpy as np
import os
from daal4py.oneapi import sycl_context, sycl_buffer

# let's try to use pandas' fast csv reader
try:
    import pandas
    read_csv = lambda f, c, t=np.float64: pandas.read_csv(f, usecols=c, delimiter=',', header=None, dtype=t)
except:
    # fall back to numpy loadtxt
    read_csv = lambda f, c, t=np.float64: np.loadtxt(f, usecols=c, delimiter=',', ndmin=2)


# At this moment with sycl we are working only with numpy arrays
def to_numpy(data):
    try:
        from pandas import DataFrame
        if isinstance(data, DataFrame):
            return np.ascontiguousarray(data.values)
    except:
        pass
    try:
        from scipy.sparse import csr_matrix
        if isinstance(data, csr_matrix):
            return data.toarray()
    except:
        pass
    return data


def main(readcsv=read_csv, method='defaultDense'):
    # Input data set parameters
    train_file = os.path.join('data', 'batch', 'k_nearest_neighbors_train.csv')
    predict_file  = os.path.join('data', 'batch', 'k_nearest_neighbors_test.csv')

    # Read data. Let's use 5 features per observation
    nFeatures = 5
    nClasses = 5
    train_data = readcsv(train_file, range(nFeatures))
    train_labels = readcsv(train_file, range(nFeatures, nFeatures+1))
    predict_data = readcsv(predict_file, range(nFeatures))
    predict_labels = readcsv(predict_file, range(nFeatures, nFeatures+1))

    train_data = to_numpy(train_data)
    train_labels = to_numpy(train_labels)
    predict_data = to_numpy(predict_data)

    # It is possible to specify to make the computations on GPU
    with sycl_context('gpu'):
        sycl_train_data = sycl_buffer(train_data)
        sycl_train_labels = sycl_buffer(train_labels)
        sycl_predict_data = sycl_buffer(predict_data)

        # Create an algorithm object and call compute
        train_algo = d4p.bf_knn_classification_training(nClasses=nClasses)
        train_result = train_algo.compute(sycl_train_data, sycl_train_labels)

        # Create an algorithm object and call compute
        predict_algo = d4p.bf_knn_classification_prediction()
        predict_result = predict_algo.compute(sycl_predict_data, train_result.model)

    # We expect less than 170 mispredicted values
    assert np.count_nonzero(predict_labels != predict_result.prediction) < 170

    return (predict_result, predict_labels)


if __name__ == "__main__":
    (predict_result, predict_labels) = main()    
    print("BF based KNN classification results:")
    print("Ground truth(observations #30-34):\n", predict_labels[30:35])
    print("Classification results(observations #30-34):\n", predict_result.prediction[30:35])
