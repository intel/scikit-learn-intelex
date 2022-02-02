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

import numpy as np
import logging
import random

from daal4py.sklearn import patch_sklearn
patch_sklearn()

from sklearn.metrics import pairwise_distances, roc_auc_score
from sklearn.datasets import (
    make_regression,
    load_iris,
    load_diabetes)

import sys
import pathlib
absolute_path = str(pathlib.Path(__file__).parent.absolute())
sys.path.append(absolute_path + '/../')
from _models_info import MODELS_INFO, TYPES


def get_class_name(x):
    return x.__class__.__name__


def generate_dataset(name, dtype, model_name):
    if model_name == 'LinearRegression':
        X, y = make_regression(n_samples=1000, n_features=5)
    elif name in ['blobs', 'classifier']:
        X, y = load_iris(return_X_y=True)
    elif name == 'regression':
        X, y = load_diabetes(return_X_y=True)
    else:
        raise ValueError('Unknown dataset type')
    X = np.array(X, dtype=dtype)
    y = np.array(y, dtype=dtype)
    return (X, y)


def run_patch(model_info, dtype):
    print(get_class_name(model_info['model']), dtype.__name__)
    X, y = generate_dataset(model_info['dataset'],
                            dtype,
                            get_class_name(model_info['model']))
    model = model_info['model']
    model.fit(X, y)
    logging.info('fit')
    for i in model_info['methods']:
        if i == 'predict':
            model.predict(X)
        elif i == 'predict_proba':
            model.predict_proba(X)
        elif i == 'predict_log_proba':
            model.predict_log_proba(X)
        elif i == 'decision_function':
            model.decision_function(X)
        elif i == 'fit_predict':
            model.fit_predict(X)
        elif i == 'transform':
            model.transform(X)
        elif i == 'fit_transform':
            model.fit_transform(X)
        elif i == 'kneighbors':
            model.kneighbors(X)
        elif i == 'score':
            model.score(X, y)
        else:
            raise ValueError(i + ' is wrong method')
        logging.info(i)


def run_algotithms():
    for info in MODELS_INFO:
        for t in TYPES:
            model_name = get_class_name(info['model'])
            if model_name in ['Ridge', 'LinearRegression'] and t.__name__ == 'uint32':
                continue
            run_patch(info, t)


def run_utils():
    # pairwise_distances
    for metric in ['cosine', 'correlation']:
        for t in TYPES:
            X = np.random.rand(1000)
            X = np.array(X, dtype=t)
            print('pairwise_distances', t.__name__)
            _ = pairwise_distances(X.reshape(1, -1), metric=metric)
            logging.info('pairwise_distances')
    # roc_auc_score
    for t in [np.float32, np.float64]:
        a = [random.randint(0, 1) for i in range(1000)]
        b = [random.randint(0, 1) for i in range(1000)]
        a = np.array(a, dtype=t)
        b = np.array(b, dtype=t)
        print('roc_auc_score', t.__name__)
        _ = roc_auc_score(a, b)
        logging.info('roc_auc_score')


if __name__ == '__main__':
    run_algotithms()
    run_utils()
