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

# daal4py Gradient Bossting Classification model creation from XGBoost example

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

import daal4py as d4p


def pd_read_csv(f, c=None, t=np.float64):
    return pd.read_csv(f, usecols=c, delimiter=",", header=None, dtype=t)


def main(readcsv=pd_read_csv):
    data_path = Path(__file__).parent / ".." / "daal4py" / "data" / "batch"
    train_file = data_path / "df_classification_train.csv"
    test_file = data_path / "df_classification_test.csv"

    # Data reading
    X_train = readcsv(train_file, range(3), t=np.float32)
    y_train = readcsv(train_file, range(3, 4), t=np.float32)
    X_test = readcsv(test_file, range(3), t=np.float32)
    y_test = readcsv(test_file, range(3, 4), t=np.float32)

    # Datasets creation
    xgb_train = xgb.DMatrix(X_train, label=np.array(y_train))
    xgb_test = xgb.DMatrix(X_test, label=np.array(y_test))

    # training parameters setting
    params = {
        "max_bin": 256,
        "scale_pos_weight": 2,
        "lambda_l2": 1,
        "alpha": 0.9,
        "max_depth": 6,
        "num_leaves": 2**6,
        "verbosity": 0,
        "objective": "multi:softmax",
        "learning_rate": 0.3,
        "num_class": 5,
        "n_estimators": 25,
    }

    # Training
    xgb_model = xgb.train(params, xgb_train, num_boost_round=100)

    # XGBoost prediction
    xgb_prediction = xgb_model.predict(xgb_test)
    xgb_errors_count = np.count_nonzero(xgb_prediction - np.ravel(y_test))

    # Conversion to daal4py
    daal_model = d4p.mb.convert_model(xgb_model)

    # daal4py prediction
    daal_prediction = daal_model.predict(X_test)
    daal_errors_count = np.count_nonzero(daal_prediction - np.ravel(y_test))
    assert np.absolute(xgb_errors_count - daal_errors_count) == 0

    return (
        xgb_prediction,
        xgb_errors_count,
        daal_prediction,
        daal_errors_count,
        np.ravel(y_test),
    )


if __name__ == "__main__":
    (
        xgb_prediction,
        xgb_errors_count,
        daal_prediction,
        daal_errors_count,
        y_test,
    ) = main()
    print("\nXGBoost prediction results (first 10 rows):\n", xgb_prediction[0:10])
    print("\ndaal4py prediction results (first 10 rows):\n", daal_prediction[0:10])
    print("\nGround truth (first 10 rows):\n", y_test[0:10])

    print("XGBoost errors count:", xgb_errors_count)
    print("XGBoost accuracy score:", 1 - xgb_errors_count / xgb_prediction.shape[0])

    print("\ndaal4py errors count:", daal_errors_count)
    print("daal4py accuracy score:", 1 - daal_errors_count / daal_prediction.shape[0])
    print("\nAll looks good!")
