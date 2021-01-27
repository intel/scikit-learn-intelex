#===============================================================================
# Copyright 2020-2021 Intel Corporation
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

# daal4py SVM example for shared memory systems

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


# Common code for both CPU and GPU computations
def compute(train_indep_data, train_dep_data, test_indep_data, method='defaultDense'):
    # Configure a SVM object to use linear kernel
    kernel_function = d4p.kernel_function_linear(
        fptype='float', method='defaultDense', k=1.0, b=0.0
    )
    train_algo = d4p.svm_training(
        fptype='float',
        method=method,
        kernel=kernel_function,
        C=1.0,
        accuracyThreshold=1e-3,
        tau=1e-8,
        cacheSize=600000000
    )

    train_result = train_algo.compute(train_indep_data, train_dep_data)

    # Create an algorithm object and call compute
    predict_algo = d4p.svm_prediction(fptype='float', kernel=kernel_function)
    predict_result = predict_algo.compute(test_indep_data, train_result.model)
    decision_result = predict_result.prediction
    predict_labels = np.where(decision_result >= 0, 1, -1)
    return predict_labels, decision_result


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


def main(readcsv=read_csv):
    # input data file
    train_file = os.path.join('..', 'data', 'batch', 'svm_two_class_train_dense.csv')
    predict_file = os.path.join('..', 'data', 'batch', 'svm_two_class_test_dense.csv')

    nFeatures = 20
    train_data = readcsv(train_file, range(nFeatures), t=np.float32)
    train_labels = readcsv(train_file, range(nFeatures, nFeatures + 1), t=np.float32)
    predict_data = readcsv(predict_file, range(nFeatures), t=np.float32)
    predict_labels = readcsv(predict_file, range(nFeatures, nFeatures + 1), t=np.float32)

    predict_result_classic, decision_function_classic = \
        compute(train_data, train_labels, predict_data, 'boser')

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

    # It is possible to specify to make the computations on GPU
    if gpu_available:
        with gpu_context():
            sycl_train_data = sycl_buffer(train_data)
            sycl_train_labels = sycl_buffer(train_labels)
            sycl_predict_data = sycl_buffer(predict_data)

            predict_result_gpu, decision_function_gpu = \
                compute(sycl_train_data, sycl_train_labels, sycl_predict_data, 'thunder')
            assert np.allclose(predict_result_gpu, predict_result_classic)

    with cpu_context():
        sycl_train_data = sycl_buffer(train_data)
        sycl_predict_data = sycl_buffer(predict_data)

        predict_result_cpu, decision_function_cpu = \
            compute(sycl_train_data, train_labels, sycl_predict_data, 'thunder')
        assert np.allclose(predict_result_cpu, predict_result_classic)

    return predict_labels, predict_result_classic, decision_function_classic


if __name__ == "__main__":
    predict_labels, predict_result, decision_function = main()
    np.set_printoptions(precision=0)
    print(
        "\nSVM classification decision function (first 10 observations):\n",
        decision_function[0:10]
    )
    print(
        "\nSVM classification predict result (first 10 observations):\n",
        predict_result[0:10]
    )
    print("\nGround truth (first 10 observations):\n", predict_labels[0:10])
    print('All looks good!')
