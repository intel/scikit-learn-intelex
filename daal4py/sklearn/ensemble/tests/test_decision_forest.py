#
# ****************************************************************************
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
# ****************************************************************************

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

N_TRIES = 10
ACCURACY_RATIO = 0.8
MSE_RATIO = 1.4
LOG_LOSS_RATIO = 2.1
ROC_AUC_RATIO = 0.97
IRIS = load_iris()
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


def check_accuracy_classifier_class_weight_iris(weight):
    for _ in range(N_TRIES):
        x_train, x_test, y_train, y_test = \
            train_test_split(IRIS.data, IRIS.target,
                             test_size=0.33, random_state=31)

        scikit_model = ScikitRandomForestClassifier(n_estimators=100,
                                                    class_weight=weight,
                                                    max_depth=5,
                                                    random_state=777)
        daal4py_model = DaalRandomForestClassifier(n_estimators=100,
                                                   class_weight=weight,
                                                   max_depth=5,
                                                   random_state=777)

        scikit_predict = scikit_model.fit(x_train, y_train).predict(x_test)
        daal4py_predict = daal4py_model.fit(x_train, y_train).predict(x_test)

        scikit_accuracy = accuracy_score(scikit_predict, y_test)
        daal4py_accuracy = accuracy_score(daal4py_predict, y_test)

        ratio = daal4py_accuracy / scikit_accuracy
        assert ratio >= ACCURACY_RATIO, \
        f'Classifier class weight: scikit_accuracy={scikit_accuracy}, daal4py_accuracy={daal4py_accuracy}'


@pytest.mark.parametrize('weight', CLASS_WEIGHTS_IRIS)
def test_accuracy_classifier_class_weight_iris(weight):
    check_accuracy_classifier_class_weight_iris(weight)


def check_log_loss_classifier_class_weight_iris(weight):
    for _ in range(N_TRIES):
        x_train, x_test, y_train, y_test = \
            train_test_split(IRIS.data, IRIS.target,
                             test_size=0.33, random_state=31)

        scikit_model = ScikitRandomForestClassifier(n_estimators=100,
                                                    class_weight=weight,
                                                    max_depth=5,
                                                    random_state=777)
        daal4py_model = DaalRandomForestClassifier(n_estimators=100,
                                                   class_weight=weight,
                                                   max_depth=5,
                                                   random_state=777)

        scikit_predict_proba = scikit_model.fit(x_train, y_train).predict_proba(x_test)
        daal4py_predict_proba = daal4py_model.fit(x_train, y_train).predict_proba(x_test)

        scikit_log_loss = log_loss(y_test, scikit_predict_proba)
        daal4py_log_loss = log_loss(y_test, daal4py_predict_proba)

        ratio = daal4py_log_loss / scikit_log_loss
        assert ratio <= LOG_LOSS_RATIO, \
        f'Classifier class weight: scikit_log_loss={scikit_log_loss}, daal4py_log_loss={daal4py_log_loss}'


@pytest.mark.parametrize('weight', CLASS_WEIGHTS_IRIS)
def test_log_loss_classifier_class_weight_iris(weight):
    if weight != {0: 0, 1: 0, 2: 0}:
        check_log_loss_classifier_class_weight_iris(weight)


def check_roc_auc_classifier_class_weight_iris(weight):
    for _ in range(N_TRIES):
        x_train, x_test, y_train, y_test = \
            train_test_split(IRIS.data, IRIS.target,
                             test_size=0.33, random_state=31)

        scikit_model = ScikitRandomForestClassifier(n_estimators=100,
                                                    class_weight=weight,
                                                    max_depth=5,
                                                    random_state=777)
        daal4py_model = DaalRandomForestClassifier(n_estimators=100,
                                                   class_weight=weight,
                                                   max_depth=5,
                                                   random_state=777)

        scikit_predict_proba = scikit_model.fit(x_train, y_train).predict_proba(x_test)
        daal4py_predict_proba = daal4py_model.fit(x_train, y_train).predict_proba(x_test)

        scikit_roc_auc = roc_auc_score(y_test, scikit_predict_proba, multi_class='ovr')
        daal4py_roc_auc = roc_auc_score(y_test, daal4py_predict_proba, multi_class='ovr')

        ratio = daal4py_roc_auc / scikit_roc_auc
        assert ratio >= ROC_AUC_RATIO, \
        f'Classifier class weight: scikit_roc_auc={scikit_roc_auc}, daal4py_roc_auc={daal4py_roc_auc}'


@pytest.mark.parametrize('weight', CLASS_WEIGHTS_IRIS)
def test_roc_auc_classifier_class_weight_iris(weight):
    if weight != {0: 0, 1: 0, 2: 0}:
        check_roc_auc_classifier_class_weight_iris(weight)

SAMPLE_WEIGHTS_IRIS = [
    (np.full_like(range(150), 0), 'Only 0'),
    (np.full_like(range(150), 1), 'Only 1'),
    (np.full_like(range(150), 5), 'Only 5'),
    (np.full_like(range(150), 50), 'Only 50'),
    (np.random.rand(150), 'Uniform distribution'),
    (np.random.normal(1000, 10, 150), 'Gaussian distribution'),
    (np.random.exponential(5, 150), 'Exponential distribution'),
    (np.random.poisson(lam=10, size=150), 'Poisson distribution'),
    (np.random.rayleigh(scale=1, size=150), 'Rayleigh distribution'),
]


def check_accuracy_classifier_sample_weight(weight):
    for _ in range(N_TRIES):
        x_train, x_test, y_train, y_test = \
            train_test_split(IRIS.data, IRIS.target, 
                             test_size=0.33, random_state=31)

        scikit_model = ScikitRandomForestClassifier(n_estimators=100,
                                                    max_depth=5,
                                                    random_state=777)
        daal4py_model = DaalRandomForestClassifier(n_estimators=100,
                                                   max_depth=5,
                                                   random_state=777)

        scikit_predict = scikit_model.fit(x_train, y_train,
                                sample_weight=weight[0][:100]).predict(x_test)
        daal4py_predict = daal4py_model.fit(x_train, y_train,
                                sample_weight=weight[0][:100]).predict(x_test)

        scikit_accuracy = accuracy_score(scikit_predict, y_test)
        daal4py_accuracy = accuracy_score(daal4py_predict, y_test)
        ratio = daal4py_accuracy / scikit_accuracy
        assert ratio >= ACCURACY_RATIO, \
        f'Classifier sample weights: sample_weight_type={weight[1]},scikit_accuracy={scikit_accuracy}, daal4py_accuracy={daal4py_accuracy}'


@pytest.mark.parametrize('weight', SAMPLE_WEIGHTS_IRIS)
def test_accuracy_classifier_sample_weight_iris(weight):
    check_accuracy_classifier_sample_weight(weight)


def check_log_loss_classifier_sample_weight(weight):
    for _ in range(N_TRIES):
        x_train, x_test, y_train, y_test = \
            train_test_split(IRIS.data, IRIS.target,
                             test_size=0.33, random_state=31)

        scikit_model = ScikitRandomForestClassifier(n_estimators=100,
                                                    max_depth=5,
                                                    random_state=777)
        daal4py_model = DaalRandomForestClassifier(n_estimators=100,
                                                   max_depth=5,
                                                   random_state=777)

        scikit_predict_proba = scikit_model.fit(x_train, y_train,
                                sample_weight=weight[0][:100]).predict_proba(x_test)
        daal4py_predict_proba = daal4py_model.fit(x_train, y_train,
                                sample_weight=weight[0][:100]).predict_proba(x_test)

        scikit_log_loss = log_loss(y_test, scikit_predict_proba, sample_weight=weight[0][100:150])
        daal4py_log_loss = log_loss(y_test, daal4py_predict_proba, sample_weight=weight[0][100:150])
        ratio = daal4py_log_loss / scikit_log_loss
        assert ratio <= LOG_LOSS_RATIO, \
        f'Classifier sample weights: sample_weight_type={weight[1]},scikit_log_loss={scikit_log_loss}, daal4py_log_loss={daal4py_log_loss}'


@pytest.mark.parametrize('weight', SAMPLE_WEIGHTS_IRIS)
def test_log_loss_classifier_sample_weight_iris(weight):
    if weight[1] != 'Only 0':
        check_log_loss_classifier_sample_weight(weight)


def check_roc_auc_classifier_sample_weight(weight):
    for _ in range(N_TRIES):
        x_train, x_test, y_train, y_test = \
            train_test_split(IRIS.data, IRIS.target, 
                             test_size=0.33, random_state=31)

        scikit_model = ScikitRandomForestClassifier(n_estimators=100,
                                                    max_depth=5,
                                                    random_state=777)
        daal4py_model = DaalRandomForestClassifier(n_estimators=100,
                                                   max_depth=5,
                                                   random_state=777)

        scikit_predict_proba = scikit_model.fit(x_train, y_train,
                                sample_weight=weight[0][:100]).predict_proba(x_test)
        daal4py_predict_proba = daal4py_model.fit(x_train, y_train,
                                sample_weight=weight[0][:100]).predict_proba(x_test)

        scikit_roc_auc = roc_auc_score(y_test, scikit_predict_proba,
                                        sample_weight=weight[0][100:150],
                                        multi_class='ovr')
        daal4py_roc_auc = roc_auc_score(y_test, daal4py_predict_proba,
                                         sample_weight=weight[0][100:150],
                                         multi_class='ovr')
        ratio = daal4py_roc_auc / scikit_roc_auc
        assert ratio >= ROC_AUC_RATIO, \
        f'Classifier sample weights: sample_weight_type={weight[1]},scikit_log_loss={scikit_log_loss}, daal4py_log_loss={daal4py_log_loss}'


@pytest.mark.parametrize('weight', SAMPLE_WEIGHTS_IRIS)
def test_roc_auc_classifier_sample_weight_iris(weight):
    if weight[1] != 'Only 0':
        check_roc_auc_classifier_sample_weight(weight)


def check_accuracy_regressor_sample_weight(weight):
    for _ in range(N_TRIES):
        x_train, x_test, y_train, y_test = \
            train_test_split(IRIS.data, IRIS.target,
                             test_size=0.33, random_state=31)

        scikit_model = ScikitRandomForestClassifier(n_estimators=100,
                                                    max_depth=5,
                                                    random_state=777)
        daal4py_model = DaalRandomForestClassifier(n_estimators=100,
                                                   max_depth=5,
                                                   random_state=777)

        scikit_predict = scikit_model.fit(x_train, y_train,
                                sample_weight=weight[0][:100]).predict(x_test)
        daal4py_predict = daal4py_model.fit(x_train, y_train,
                                sample_weight=weight[0][:100]).predict(x_test)

        scikit_mse = mean_squared_error(scikit_predict, y_test)
        daal4py_mse = mean_squared_error(daal4py_predict, y_test)

        ratio = daal4py_mse / scikit_mse
        assert ratio <= MSE_RATIO, \
        f'Regression sample weights: sample_weight_type={weight[1]},scikit_mse={scikit_mse}, daal4py_mse={daal4py_mse}'


@pytest.mark.parametrize('weight', SAMPLE_WEIGHTS_IRIS)
def test_accuracy_regressor_sample_weight_iris(weight):
    if weight[1] != 'Only 0':
        check_accuracy_regressor_sample_weight(weight)
