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

# daal4py BF KNN example for shared memory systems

import daal4py as d4p
import numpy as np
import os
from daal4py.oneapi import sycl_buffer

# let's try to use pandas' fast csv reader
try:
    import pandas

    def read_csv(f, c, t=np.float64):
        return pandas.read_csv(f, usecols=c, delimiter=',', header=None, dtype=t)
except ImportError:
    # fall back to numpy loadtxt
    def read_csv(f, c, t=np.float64):
        return np.loadtxt(f, usecols=c, delimiter=',', ndmin=2)

try:
    from dpctx import device_context, device_type
    with device_context(device_type.gpu, 0):
        gpu_available = True
except:
    try:
        from daal4py.oneapi import sycl_context
        with sycl_context('gpu'):
            gpu_available = True
    except:
        gpu_available = False


# At this moment with sycl we are working only with numpy arrays
def to_numpy(data):
    try:
        from pandas import DataFrame
        if isinstance(data, DataFrame):
            return np.ascontiguousarray(data.values)
    except ImportError:
        pass
    try:
        from scipy.sparse import csr_matrix
        if isinstance(data, csr_matrix):
            return data.toarray()
    except ImportError:
        pass
    return data


# Common code for both CPU and GPU computations
def compute(train_data, train_labels, predict_data, nClasses):
    # Create an algorithm object and call compute
    train_algo = d4p.bf_knn_classification_training(nClasses=nClasses, fptype='float')
    train_result = train_algo.compute(train_data, train_labels)

    # Create an algorithm object and call compute
    predict_algo = d4p.bf_knn_classification_prediction(nClasses=nClasses, fptype='float')
    predict_result = predict_algo.compute(predict_data, train_result.model)
    return predict_result


def main(readcsv=read_csv, method='defaultDense'):
    # Input data set parameters
    train_file = os.path.join('..', 'data', 'batch', 'k_nearest_neighbors_train.csv')
    predict_file = os.path.join('..', 'data', 'batch', 'k_nearest_neighbors_test.csv')

    # Read data. Let's use 5 features per observation
    nFeatures = 5
    nClasses = 5
    train_data = readcsv(train_file, range(nFeatures), t=np.float32)
    train_labels = readcsv(train_file, range(nFeatures, nFeatures + 1), t=np.float32)
    predict_data = readcsv(predict_file, range(nFeatures), t=np.float32)
    predict_labels = readcsv(predict_file, range(nFeatures, nFeatures + 1), t=np.float32)

    predict_result_classic = compute(train_data, train_labels, predict_data, nClasses)

    # We expect less than 170 mispredicted values
    assert np.count_nonzero(predict_labels != predict_result_classic.prediction) < 170

    train_data = to_numpy(train_data)
    train_labels = to_numpy(train_labels)
    predict_data = to_numpy(predict_data)

    try:
        from dpctx import device_context, device_type

        def gpu_context():
            return device_context(device_type.gpu, 0)

        def cpu_context():
            return device_context(device_type.cpu, 0)
    except:
        from daal4py.oneapi import sycl_context

        def gpu_context():
            return sycl_context('gpu')

        def cpu_context():
            return sycl_context('cpu')

    if gpu_available:
        with gpu_context():
            sycl_train_data = sycl_buffer(train_data)
            sycl_train_labels = sycl_buffer(train_labels)
            sycl_predict_data = sycl_buffer(predict_data)

            predict_result_gpu = compute(sycl_train_data, sycl_train_labels,
                                         sycl_predict_data, nClasses)
            assert np.allclose(predict_result_gpu.prediction,
                               predict_result_classic.prediction)

    with cpu_context():
        sycl_train_data = sycl_buffer(train_data)
        sycl_train_labels = sycl_buffer(train_labels)
        sycl_predict_data = sycl_buffer(predict_data)

        predict_result_cpu = compute(sycl_train_data, sycl_train_labels,
                                     sycl_predict_data, nClasses)
        assert np.allclose(predict_result_cpu.prediction,
                           predict_result_classic.prediction)

    return (predict_result_classic, predict_labels)


if __name__ == "__main__":
    (predict_result, predict_labels) = main()
    print("BF based KNN classification results:")
    print("Ground truth(observations #30-34):\n", predict_labels[30:35])
    print(
        "Classification results(observations #30-34):\n",
        predict_result.prediction[30:35]
    )
