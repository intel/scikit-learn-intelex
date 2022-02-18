#===============================================================================
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
#===============================================================================

# daal4py Gradient Bossting Classification model creation from Catboost example

import daal4py as d4p
import catboost as cb
import numpy as np
import pandas as pd


def pd_read_csv(f, c=None, t=np.float64):
    return pd.read_csv(f, usecols=c, delimiter=',', header=None, dtype=t)


def main(readcsv=pd_read_csv, method='defaultDense'):
    # Path to data
    train_file = "./data/batch/df_classification_train.csv"
    test_file = "./data/batch/df_classification_test.csv"

    # Data reading
    X_train = readcsv(train_file, range(3), t=np.float32)
    y_train = readcsv(train_file, range(3, 4), t=np.float32)
    X_test = readcsv(test_file, range(3), t=np.float32)
    y_test = readcsv(test_file, range(3, 4), t=np.float32)

    # Datasets creation
    cb_train = cb.Pool(X_train, label=np.array(y_train))
    cb_test = cb.Pool(X_test, label=np.array(y_test))

    # training parameters setting
    params = {
        'reg_lambda': 1,
        'max_depth': 8,
        'num_leaves': 2**8,
        'verbose': 0,
        'objective': 'MultiClass',
        'learning_rate': 0.3,
        'n_estimators': 100,
        'classes_count': 5,
    }

    # Training
    cb_model = cb.CatBoost(params)
    cb_model.fit(cb_train)

    # Catboost prediction
    cb_prediction = cb_model.predict(cb_test, prediction_type='Class').T[0]
    cb_errors_count = np.count_nonzero(cb_prediction - np.ravel(y_test))

    # Conversion to daal4py
    daal_model = d4p.get_gbt_model_from_catboost(cb_model)

    # daal4py prediction
    daal_predict_algo = d4p.gbt_classification_prediction(
        nClasses=params['classes_count'],
        resultsToEvaluate="computeClassLabels",
        fptype='float'
    )
    daal_prediction = daal_predict_algo.compute(X_test, daal_model)
    daal_errors_count = np.count_nonzero(daal_prediction.prediction - y_test)
    assert np.absolute(cb_errors_count - daal_errors_count) == 0

    return (cb_prediction, cb_errors_count, np.ravel(daal_prediction.prediction),
            daal_errors_count, np.ravel(y_test))


if __name__ == "__main__":
    (cb_prediction, cb_errors_count,
     daal_prediction, daal_errors_count, y_test) = main()
    print("\nCatboost prediction results (first 10 rows):\n",
          cb_prediction[0:10])
    print("\ndaal4py prediction results (first 10 rows):\n",
          daal_prediction[0:10])
    print("\nGround truth (first 10 rows):\n", y_test[0:10])

    print("Catboost errors count:", cb_errors_count)
    print("Catboost accuracy score:",
          1 - cb_errors_count / cb_prediction.shape[0])

    print("\ndaal4py errors count:", daal_errors_count)
    print("daal4py accuracy score:",
          1 - daal_errors_count / daal_prediction.shape[0])
    print("\nAll looks good!")
