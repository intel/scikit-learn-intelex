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

# daal4py Gradient Bossting Classification model creation from LightGBM example

import daal4py as d4p
import lightgbm as lgb
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
    lgb_train = lgb.Dataset(
        X_train,
        np.array(y_train).reshape(X_train.shape[0]),
        free_raw_data=False
    )

    # training parameters setting
    params = {
        'max_bin': 256,
        'scale_pos_weight': 2,
        'lambda_l2': 1,
        'alpha': 0.9,
        'max_depth': 8,
        'num_leaves': 2**8,
        'verbose': -1,
        'objective': 'multiclass',
        'learning_rate': 0.3,
        'num_class': 5,
    }

    # Training
    lgb_model = lgb.train(params, lgb_train, valid_sets=lgb_train, verbose_eval=False)

    # LightGBM prediction
    lgb_prediction = np.argmax(lgb_model.predict(X_test), axis=1)
    lgb_errors_count = np.count_nonzero(lgb_prediction - np.ravel(y_test))

    # Conversion to daal4py
    daal_model = d4p.get_gbt_model_from_lightgbm(lgb_model)

    # daal4py prediction
    daal_predict_algo = d4p.gbt_classification_prediction(
        nClasses=params["num_class"],
        resultsToEvaluate="computeClassLabels",
        fptype='float'
    )
    daal_prediction = daal_predict_algo.compute(X_test, daal_model)
    daal_errors_count = np.count_nonzero(daal_prediction.prediction - y_test)
    assert np.absolute(lgb_errors_count - daal_errors_count) == 0

    return (lgb_prediction, lgb_errors_count, np.ravel(daal_prediction.prediction),
            daal_errors_count, np.ravel(y_test))


if __name__ == "__main__":
    (lgb_prediction, lgb_errors_count, daal_prediction,
     daal_errors_count, y_test) = main()
    print("\nLightGBM prediction results (first 10 rows):\n", lgb_prediction[0:10])
    print("\ndaal4py prediction results (first 10 rows):\n", daal_prediction[0:10])
    print("\nGround truth (first 10 rows):\n", y_test[0:10])

    print("LightGBM errors count:", lgb_errors_count)
    print("LightGBM accuracy score:", 1 - lgb_errors_count / lgb_prediction.shape[0])

    print("\ndaal4py errors count:", daal_errors_count)
    print("daal4py accuracy score:", 1 - daal_errors_count / daal_prediction.shape[0])
    print("\nAll looks good!")
