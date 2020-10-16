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

from daal4py.sklearn._utils import daal_check_version
import numpy as np
import pytest
import random
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.ensemble \
    import RandomForestClassifier as ScikitRandomForestClassifier
from daal4py.sklearn.ensemble \
    import RandomForestClassifier as DaalRandomForestClassifier
from sklearn.ensemble \
    import RandomForestRegressor as ScikitRandomForestRegressor
from daal4py.sklearn.ensemble \
    import RandomForestRegressor as DaalRandomForestRegressor
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

N_TRIES = 10
CHECK_RATIO_CLASSIFIER = 0.85
CHECK_RATIO_REGRESSOR = 1.3
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


def check_classifier_class_weight_iris(weight):
    for _ in range(N_TRIES):
        x_train, x_test, y_train, y_test = \
            train_test_split(IRIS.data, IRIS.target, 
                             test_size=0.3, random_state=31)

        scikit_model = ScikitRandomForestClassifier(class_weight=weight, 
                                                    random_state=777)
        daal4py_model = DaalRandomForestClassifier(class_weight=weight, 
                                                   random_state=777)

        scikit_predict = scikit_model.fit(x_train, y_train).predict(x_test)
        daal4py_predict = daal4py_model.fit(x_train, y_train).predict(x_test)

        scikit_accuracy = accuracy_score(scikit_predict, y_test)
        daal4py_accuracy = accuracy_score(daal4py_predict, y_test)

        ratio = daal4py_accuracy / scikit_accuracy
        assert ratio >= CHECK_RATIO_CLASSIFIER, ('Classifier class weight: scikit_accuracy=%f,\
            daal4py_accuracy=%f' % (scikit_accuracy, daal4py_accuracy))


@pytest.mark.parametrize('weight', CLASS_WEIGHTS_IRIS)
def test_classifier_class_weight_iris(weight):
    if daal_check_version((2021,'B', 110)):
        check_classifier_class_weight_iris(weight)

SAMPLE_WEIGHTS_IRIS = [
    (np.full_like(range(100), 0), 'Only 0'),
    (np.full_like(range(100), 1), 'Only 1'),
    (np.full_like(range(100), 5), 'Only 5'),
    (np.full_like(range(100), 50), 'Only 50'),
    (np.random.rand(100), 'Uniform distribution'),
    (np.random.normal(1000, 10, 100), 'Gaussian distribution'),
    (np.random.exponential(5, 100), 'Exponential distribution'),
    (np.random.poisson(lam=10, size=100), 'Poisson distribution'),
    (np.random.rayleigh(scale=1, size=100), 'Rayleigh distribution'),
]


def check_classifier_sample_weight(weight, check_ratio=0.9):
    for _ in range(N_TRIES):
        x_train, x_test, y_train, y_test = \
            train_test_split(IRIS.data, IRIS.target, 
                             test_size=0.33, random_state=31)

        scikit_model = ScikitRandomForestClassifier(random_state=777)
        daal4py_model = DaalRandomForestClassifier(random_state=777)

        scikit_predict = scikit_model.fit(x_train, y_train,
                                  sample_weight=weight[0]).predict(x_test)
        daal4py_predict = daal4py_model.fit(x_train, y_train,
                                    sample_weight=weight[0]).predict(x_test)

        scikit_accuracy = accuracy_score(scikit_predict, y_test)
        daal4py_accuracy = accuracy_score(daal4py_predict, y_test)
        ratio = daal4py_accuracy / scikit_accuracy
        assert ratio >= CHECK_RATIO_CLASSIFIER, \
            ('Classifier sample weights: sample_weight_type=%s,scikit_accuracy=%f,\
            daal4py_accuracy=%f' % (weight[1], scikit_accuracy, daal4py_accuracy))


@pytest.mark.parametrize('weight', SAMPLE_WEIGHTS_IRIS)
def test_classifier_sample_weight_iris(weight):
    if daal_check_version((2021,'B', 110)):
        check_classifier_sample_weight(weight)


def check_regressor_sample_weight(weight, check_ratio=1.1):
    for _ in range(N_TRIES):
        x_train, x_test, y_train, y_test = \
            train_test_split(IRIS.data, IRIS.target, 
                             test_size=0.33, random_state=31)

        scikit_model = ScikitRandomForestRegressor(random_state=777)
        daal4py_model = DaalRandomForestRegressor(random_state=777)

        scikit_predict = scikit_model.fit(x_train, y_train,
                                  sample_weight=weight[0]).predict(x_test)
        daal4py_predict = daal4py_model.fit(x_train, y_train,
                                    sample_weight=weight[0]).predict(x_test)

        scikit_accuracy = mean_squared_error(scikit_predict, y_test)
        daal4py_accuracy = mean_squared_error(daal4py_predict, y_test)
        ratio = daal4py_accuracy / scikit_accuracy
        assert ratio <= CHECK_RATIO_REGRESSOR, \
            ('Regression sample weights: sample_weight_type=%s,scikit_accuracy=%f,\
            daal4py_accuracy=%f' % (weight[1], scikit_accuracy, daal4py_accuracy))


@pytest.mark.parametrize('weight', SAMPLE_WEIGHTS_IRIS)
def test_regressor_sample_weight_iris(weight):
    if daal_check_version((2021,'B', 110)) and weight[1] != 'Only 0':
        check_regressor_sample_weight(weight)
