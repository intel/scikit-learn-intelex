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

import numpy as np
import pytest
import random
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             mean_squared_error, log_loss)
from sklearn.ensemble \
    import RandomForestClassifier as ScikitRandomForestClassifier
from daal4py.sklearn.ensemble \
    import RandomForestClassifier as DaalRandomForestClassifier
from sklearn.ensemble \
    import RandomForestRegressor as ScikitRandomForestRegressor
from daal4py.sklearn.ensemble \
    import RandomForestRegressor as DaalRandomForestRegressor
from daal4py.sklearn._utils import daal_check_version

N_TRIES = 10
ACCURACY_RATIO = 0.85 if daal_check_version((2021, 'P', 200)) else 0.7
MSE_RATIO = 1.05 if daal_check_version((2021, 'P', 200)) else 1.42
LOG_LOSS_RATIO = 1.55 if daal_check_version((2021, 'P', 200)) else 2.28
ROC_AUC_RATIO = 0.978
IRIS = load_iris()

random.seed(777)
CLASS_WEIGHTS_IRIS = [
    {0: 0, 1: 0, 2: 0},
    {0: 0, 1: 1, 2: 1},
    {0: 1, 1: 2, 2: 3},
    {0: 10, 1: 5, 2: 4},
    {
        0: random.uniform(1, 50),
        1: random.uniform(1, 50),
        2: random.uniform(1, 50),
    },
    {
        0: random.uniform(50, 100),
        1: random.uniform(50, 100),
        2: random.uniform(50, 100),
    },
    {
        0: random.uniform(1, 1000),
        1: random.uniform(1, 1000),
        2: random.uniform(1, 1000),
    },
    {
        0: random.uniform(1, 10),
        1: random.uniform(50, 100),
        2: random.uniform(1, 100),
    },
    {
        0: random.uniform(1, 10),
        1: random.uniform(1, 100),
        2: random.uniform(1, 1000),
    },
    {
        0: random.uniform(1, 2000),
        1: random.uniform(1, 2000),
        2: random.uniform(1, 2000),
    },
    {0: 50, 1: 50, 2: 50},
    'balanced',
]


def _test_classifier_class_weight_iris(weight):
    for _ in range(N_TRIES):
        x_train, x_test, y_train, y_test = \
            train_test_split(IRIS.data, IRIS.target,
                             test_size=0.33, random_state=31)
        # models
        scikit_model = ScikitRandomForestClassifier(n_estimators=100,
                                                    class_weight=weight,
                                                    max_depth=None,
                                                    random_state=777)
        daal4py_model = DaalRandomForestClassifier(n_estimators=100,
                                                   class_weight=weight,
                                                   max_depth=None,
                                                   random_state=777)
        # training
        scikit_model.fit(x_train, y_train)
        daal4py_model.fit(x_train, y_train)
        #predict
        scikit_predict = scikit_model.predict(x_test)
        daal4py_predict = daal4py_model.predict(x_test)
        #accuracy
        scikit_accuracy = accuracy_score(scikit_predict, y_test)
        daal4py_accuracy = accuracy_score(daal4py_predict, y_test)
        ratio = daal4py_accuracy / scikit_accuracy
        reason = ("Classifier class weight: scikit_accuracy={},"
                  "daal4py_accuracy={}".format(
                      scikit_accuracy, daal4py_accuracy))
        assert ratio >= ACCURACY_RATIO, reason

        if weight == {0: 0, 1: 0, 2: 0}:
            continue
        # predict_proba
        scikit_predict_proba = scikit_model.predict_proba(x_test)
        daal4py_predict_proba = daal4py_model.predict_proba(x_test)
        #log_loss
        scikit_log_loss = log_loss(y_test, scikit_predict_proba)
        daal4py_log_loss = log_loss(y_test, daal4py_predict_proba)
        ratio = daal4py_log_loss / scikit_log_loss
        reason = ("Classifier class weight: scikit_log_loss="
                  "{}, daal4py_log_loss={}".format(
                      scikit_log_loss, daal4py_log_loss))
        assert ratio <= LOG_LOSS_RATIO, reason

        # ROC AUC
        scikit_roc_auc = roc_auc_score(y_test, scikit_predict_proba, multi_class='ovr')
        daal4py_roc_auc = roc_auc_score(y_test, daal4py_predict_proba, multi_class='ovr')
        ratio = daal4py_roc_auc / scikit_roc_auc
        reason = "Classifier class weight: scikit_roc_auc={}, daal4py_roc_auc={}".format(
            scikit_roc_auc, daal4py_roc_auc)
        assert ratio >= ROC_AUC_RATIO, reason


@pytest.mark.parametrize('weight', CLASS_WEIGHTS_IRIS)
def test_classifier_class_weight_iris(weight):
    _test_classifier_class_weight_iris(weight)


np.random.seed(777)
SAMPLE_WEIGHTS_IRIS = [
    (np.full_like(range(150), 0), 'Only 0'),
    (np.full_like(range(150), 1), 'Only 1'),
    (np.full_like(range(150), 5), 'Only 5'),
    (np.full_like(range(150), 50), 'Only 50'),
    (np.random.rand(150), 'Uniform distribution'),
    (np.random.normal(1000, 10, 150), 'Gaussian distribution'),
    (np.random.poisson(lam=10, size=150), 'Poisson distribution'),
    (np.random.rayleigh(scale=1, size=150), 'Rayleigh distribution'),
]


def _test_classifier_sample_weight(weight):
    for _ in range(N_TRIES):
        x_train, x_test, y_train, y_test = \
            train_test_split(IRIS.data, IRIS.target,
                             test_size=0.33, random_state=31)
        #models
        scikit_model = ScikitRandomForestClassifier(n_estimators=100,
                                                    max_depth=None,
                                                    random_state=777)
        daal4py_model = DaalRandomForestClassifier(n_estimators=100,
                                                   max_depth=None,
                                                   random_state=777)
        #training
        scikit_model.fit(x_train, y_train,
                         sample_weight=weight[0][:100])
        daal4py_model.fit(x_train, y_train,
                          sample_weight=weight[0][:100])
        #predict
        scikit_predict = scikit_model.predict(x_test)
        daal4py_predict = daal4py_model.predict(x_test)
        #accuracy
        scikit_accuracy = accuracy_score(scikit_predict, y_test)
        daal4py_accuracy = accuracy_score(daal4py_predict, y_test)
        ratio = daal4py_accuracy / scikit_accuracy
        reason = ("Classifier sample weights: sample_weight_type={},"
                  "scikit_accuracy={}, daal4py_accuracy={}".format(
                      weight[1], scikit_accuracy, daal4py_accuracy))
        assert ratio >= ACCURACY_RATIO, reason

        if weight[1] == 'Only 0':
            continue
        # predict_proba
        scikit_predict_proba = scikit_model.predict_proba(x_test)
        daal4py_predict_proba = daal4py_model.predict_proba(x_test)
        #log_loss
        scikit_log_loss = log_loss(
            y_test, scikit_predict_proba, sample_weight=weight[0][100:150])
        daal4py_log_loss = log_loss(
            y_test, daal4py_predict_proba, sample_weight=weight[0][100:150])
        ratio = daal4py_log_loss / scikit_log_loss
        reason = ("Classifier sample weights: sample_weight_type={},"
                  "scikit_log_loss={}, daal4py_log_loss={}".format(
                      weight[1], scikit_log_loss, daal4py_log_loss))
        assert ratio <= LOG_LOSS_RATIO, reason

        # ROC AUC
        scikit_roc_auc = roc_auc_score(y_test, scikit_predict_proba,
                                       sample_weight=weight[0][100:150],
                                       multi_class='ovr')
        daal4py_roc_auc = roc_auc_score(y_test, daal4py_predict_proba,
                                        sample_weight=weight[0][100:150],
                                        multi_class='ovr')
        ratio = daal4py_roc_auc / scikit_roc_auc
        reason = ("Classifier sample weights: sample_weight_type={},"
                  "scikit_roc_auc={}, daal4py_roc_auc={}".format(
                      weight[1], scikit_roc_auc, daal4py_roc_auc))
        assert ratio >= ROC_AUC_RATIO, reason


@pytest.mark.parametrize('weight', SAMPLE_WEIGHTS_IRIS)
def test_classifier_sample_weight_iris(weight):
    _test_classifier_sample_weight(weight)


def _test_mse_regressor_sample_weight(weight):
    for _ in range(N_TRIES):
        x_train, x_test, y_train, y_test = \
            train_test_split(IRIS.data, IRIS.target,
                             test_size=0.33, random_state=31)

        scikit_model = ScikitRandomForestRegressor(n_estimators=100,
                                                   max_depth=None,
                                                   random_state=777)
        daal4py_model = DaalRandomForestRegressor(n_estimators=100,
                                                  max_depth=None,
                                                  random_state=777)

        scikit_predict = scikit_model.fit(
            x_train, y_train,
            sample_weight=weight[0][:100]).predict(x_test)
        daal4py_predict = daal4py_model.fit(
            x_train, y_train,
            sample_weight=weight[0][:100]).predict(x_test)

        scikit_mse = mean_squared_error(scikit_predict, y_test)
        daal4py_mse = mean_squared_error(daal4py_predict, y_test)

        ratio = daal4py_mse / scikit_mse
        reason = ("Regression sample weights: sample_weight_type={},"
                  "scikit_mse={}, daal4py_mse={}".format(
                      weight[1], scikit_mse, daal4py_mse))
        assert ratio <= MSE_RATIO, reason


@pytest.mark.parametrize('weight', SAMPLE_WEIGHTS_IRIS)
def test_mse_regressor_sample_weight_iris(weight):
    if weight[1] != 'Only 0':
        _test_mse_regressor_sample_weight(weight)
