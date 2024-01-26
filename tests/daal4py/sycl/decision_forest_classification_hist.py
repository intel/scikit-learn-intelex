# ==============================================================================
# Copyright 2021 Intel Corporation
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

# daal4py Decision Forest Classification example of Hist method for shared memory systems

import os
from pathlib import Path

import numpy as np

import daal4py as d4p
from daal4py.oneapi import sycl_buffer

# let's try to use pandas' fast csv reader
try:
    import pandas

    def read_csv(f, c, t=np.float64):
        return pandas.read_csv(f, usecols=c, delimiter=",", header=None, dtype=t)

except Exception:
    # fall back to numpy loadtxt
    def read_csv(f, c, t=np.float64):
        return np.loadtxt(f, usecols=c, delimiter=",", ndmin=2, dtype=t)


try:
    from daal4py.oneapi import sycl_context

    with sycl_context("gpu"):
        gpu_available = True
except Exception:
    gpu_available = False


# Commone code for both CPU and GPU computations
def compute(train_data, train_labels, predict_data):
    # Configure a training object (5 classes)
    train_algo = d4p.decision_forest_classification_training(
        5,
        fptype="float",
        method="hist",
        maxBins=256,
        minBinSize=1,
        nTrees=10,
        minObservationsInLeafNode=8,
        featuresPerNode=3,
        engine=d4p.engines_mt19937(seed=777),
        varImportance="MDI",
        bootstrap=True,
        resultsToCompute="computeOutOfBagError",
    )

    # Training result provides (depending on parameters) model,
    # outOfBagError, outOfBagErrorPerObservation and/or variableImportance
    train_result = train_algo.compute(train_data, train_labels)

    # now predict using the model from the training above
    predict_algo = d4p.decision_forest_classification_prediction(
        nClasses=5,
        fptype="float",
        resultsToEvaluate="computeClassLabels|computeClassProbabilities",
        votingMethod="unweighted",
    )

    predict_result = predict_algo.compute(predict_data, train_result.model)

    return train_result, predict_result


# At this moment with sycl we are working only with numpy arrays
def to_numpy(data):
    try:
        from pandas import DataFrame

        if isinstance(data, DataFrame):
            return np.ascontiguousarray(data.values)
    except Exception:
        try:
            from scipy.sparse import csr_matrix

            if isinstance(data, csr_matrix):
                return data.toarray()
        except Exception:
            return data

    return data


def main(readcsv=read_csv):
    nFeatures = 3
    # input data file
    data_path = Path(__file__).parent.parent / "data" / "batch"
    train_file = data_path / "df_classification_train.csv"
    predict_file = data_path / "df_classification_test.csv"

    # Read train data. Let's use 3 features per observation
    train_data = readcsv(train_file, range(nFeatures), t=np.float32)
    train_labels = readcsv(train_file, range(nFeatures, nFeatures + 1), t=np.float32)
    # Read test data (with same #features)
    predict_data = readcsv(predict_file, range(nFeatures), t=np.float32)
    predict_labels = readcsv(predict_file, range(nFeatures, nFeatures + 1), t=np.float32)

    # Using of the classic way (computations on CPU)
    train_result, predict_result = compute(train_data, train_labels, predict_data)
    assert predict_result.prediction.shape == (predict_labels.shape[0], 1)
    assert (np.mean(predict_result.prediction != predict_labels) < 0.04).any()

    train_data = to_numpy(train_data)
    train_labels = to_numpy(train_labels)
    predict_data = to_numpy(predict_data)

    # It is possible to specify to make the computations on GPU
    if gpu_available:
        with sycl_context("gpu"):
            sycl_train_data = sycl_buffer(train_data)
            sycl_train_labels = sycl_buffer(train_labels)
            sycl_predict_data = sycl_buffer(predict_data)
            train_result, predict_result = compute(
                sycl_train_data, sycl_train_labels, sycl_predict_data
            )
            assert predict_result.prediction.shape == (predict_labels.shape[0], 1)
            assert (np.mean(predict_result.prediction != predict_labels) < 0.03).any()

    return (train_result, predict_result, predict_labels)


if __name__ == "__main__":
    (train_result, predict_result, plabels) = main()
    print("\nVariable importance results:\n", train_result.variableImportance)
    print("\nOOB error:\n", train_result.outOfBagError)
    print(
        "\nDecision forest prediction results (first 10 rows):\n",
        predict_result.prediction[0:10],
    )
    print(
        "\nDecision forest probabilities results (first 10 rows):\n",
        predict_result.probabilities[0:10],
    )
    print("\nGround truth (first 10 rows):\n", plabels[0:10])
    print("All looks good!")
