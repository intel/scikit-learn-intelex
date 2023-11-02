# ==============================================================================
# Copyright 2020 Intel Corporation
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

# daal4py Gradient Boosting Classification model creation from LightGBM example

import sys
from functools import wraps
from pathlib import Path
from time import time

import lightgbm as lgb
import numpy as np
import pandas as pd

import daal4py as d4p


def pd_read_csv(f, c=None, t=np.float64):
    return pd.read_csv(f, usecols=c, delimiter=",", header=None, dtype=t)


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t = time()
        retval = func(*args, **kwargs)
        print(f"{func.__name__}(...) took {time() - t:.2f} s", file=sys.stderr)
        return retval

    return wrapper


def main(readcsv=pd_read_csv, nrows=5_000):
    data_path = Path(__file__).parent / ".." / "daal4py" / "data" / "batch"
    train_file = data_path / "df_classification_train.csv"
    test_file = data_path / "df_classification_test.csv"

    # Data reading
    X_train = readcsv(train_file, range(3), t=np.float32, nrows=nrows)
    y_train = readcsv(train_file, range(3, 4), t=np.float32, nrows=nrows)
    X_test = readcsv(test_file, range(3), t=np.float32)
    y_test = readcsv(test_file, range(3, 4), t=np.float32)

    # Datasets creation
    lgb_train = timeit(lgb.Dataset)(
        X_train, np.array(y_train).reshape(X_train.shape[0]), free_raw_data=False
    )

    # training parameters setting
    params = {
        "max_bin": 256,
        "scale_pos_weight": 2,
        "lambda_l2": 1,
        "alpha": 0.9,
        "max_depth": 6,
        "num_leaves": 2**6,
        "verbose": -1,
        "objective": "multiclass",
        "learning_rate": 0.3,
        "num_class": 5,
        "n_estimators": 25,
    }

    # Training
    lgb_model = timeit(lgb.train)(
        params, lgb_train, valid_sets=lgb_train, callbacks=[lgb.log_evaluation(0)]
    )

    # LightGBM prediction
    lgb_prediction = np.argmax(lgb_model.predict(X_test), axis=1)
    lgb_errors_count = np.count_nonzero(lgb_prediction - np.ravel(y_test))

    # Conversion to daal4py
    daal_model = timeit(d4p.mb.convert_model)(lgb_model)

    # daal4py prediction
    daal_prediction = timeit(daal_model.predict)(X_test)
    daal_errors_count = np.count_nonzero(daal_prediction - np.ravel(y_test))
    assert np.absolute(lgb_errors_count - daal_errors_count) == 0

    return (
        lgb_prediction,
        lgb_errors_count,
        daal_prediction,
        daal_errors_count,
        np.ravel(y_test),
    )


if __name__ == "__main__":
    (
        lgb_prediction,
        lgb_errors_count,
        daal_prediction,
        daal_errors_count,
        y_test,
    ) = main(nrows=None)
    print("\nLightGBM prediction results (first 10 rows):\n", lgb_prediction[0:10])
    print("\ndaal4py prediction results (first 10 rows):\n", daal_prediction[0:10])
    print("\nGround truth (first 10 rows):\n", y_test[0:10])

    print("LightGBM errors count:", lgb_errors_count)
    print("LightGBM accuracy score:", 1 - lgb_errors_count / lgb_prediction.shape[0])

    print("\ndaal4py errors count:", daal_errors_count)
    print("daal4py accuracy score:", 1 - daal_errors_count / daal_prediction.shape[0])
    print("\nAll looks good!")
