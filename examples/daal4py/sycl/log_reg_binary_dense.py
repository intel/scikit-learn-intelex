# ==============================================================================
# Copyright 2014 Intel Corporation
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
# ==============================================================================

# daal4py logistic regression example for shared memory systems

import os

import numpy as np

import daal4py as d4p
from daal4py.oneapi import sycl_buffer

# let's try to use pandas' fast csv reader
try:
    import pandas

    def read_csv(f, c, t=np.float64):
        return pandas.read_csv(f, usecols=c, delimiter=",", header=None, dtype=t)

except ImportError:
    # fall back to numpy loadtxt
    def read_csv(f, c, t=np.float64):
        return np.loadtxt(f, usecols=c, delimiter=",", ndmin=2)


try:
    from daal4py.oneapi import sycl_context

    with sycl_context("gpu"):
        gpu_available = True
except:
    gpu_available = False


# Commone code for both CPU and GPU computations
def compute(train_data, train_labels, predict_data, nClasses):
    # set parameters and train
    train_alg = d4p.logistic_regression_training(
        nClasses=nClasses, interceptFlag=True, fptype="float"
    )
    train_result = train_alg.compute(train_data, train_labels)
    # set parameters and compute predictions
    predict_alg = d4p.logistic_regression_prediction(nClasses=nClasses, fptype="float")
    return predict_alg.compute(predict_data, train_result.model), train_result


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


def main(readcsv=read_csv, method="defaultDense"):
    nClasses = 2
    nFeatures = 20

    # read training data from file with 20 features per observation and 1 class label
    trainfile = os.path.join("..", "data", "batch", "binary_cls_train.csv")
    train_data = readcsv(trainfile, range(nFeatures), t=np.float32)
    train_labels = readcsv(trainfile, range(nFeatures, nFeatures + 1), t=np.float32)

    # read testing data from file with 20 features per observation
    testfile = os.path.join("..", "data", "batch", "binary_cls_test.csv")
    predict_data = readcsv(testfile, range(nFeatures), t=np.float32)
    predict_labels = readcsv(testfile, range(nFeatures, nFeatures + 1), t=np.float32)

    # Using of the classic way (computations on CPU)
    result_classic, train_result = compute(
        train_data, train_labels, predict_data, nClasses
    )

    train_data = to_numpy(train_data)
    train_labels = to_numpy(train_labels)
    predict_data = to_numpy(predict_data)

    # It is possible to specify to make the computations on GPU
    if gpu_available:
        with sycl_context("gpu"):
            sycl_train_data = sycl_buffer(train_data)
            sycl_train_labels = sycl_buffer(train_labels)
            sycl_predict_data = sycl_buffer(predict_data)
            result_gpu, _ = compute(
                sycl_train_data, sycl_train_labels, sycl_predict_data, nClasses
            )

        # TODO: When LogisticRegression run2run instability will be replace on np.equal
        assert np.mean(result_classic.prediction != result_gpu.prediction) < 0.2

    # It is possible to specify to make the computations on GPU
    with sycl_context("cpu"):
        sycl_train_data = sycl_buffer(train_data)
        sycl_train_labels = sycl_buffer(train_labels)
        sycl_predict_data = sycl_buffer(predict_data)
        result_cpu, _ = compute(
            sycl_train_data, sycl_train_labels, sycl_predict_data, nClasses
        )

    # the prediction result provides prediction
    assert result_classic.prediction.shape == (
        predict_data.shape[0],
        train_labels.shape[1],
    )

    # TODO: When LogisticRegression run2run instability will be replace on np.equal
    assert np.mean(result_classic.prediction != result_cpu.prediction) < 0.2
    return (train_result, result_classic, predict_labels)


if __name__ == "__main__":
    (train_result, predict_result, predict_labels) = main()
    print("\nLogistic Regression coefficients:\n", train_result.model.Beta)
    print(
        "\nLogistic regression prediction results (first 10 rows):\n",
        predict_result.prediction[0:10],
    )
    print("\nGround truth (first 10 rows):\n", predict_labels[0:10])
    print("All looks good!")
