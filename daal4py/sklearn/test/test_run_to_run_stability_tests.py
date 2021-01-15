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

import daal4py as d4p
import numpy as np
import pytest

from daal4py.sklearn import patch_sklearn
patch_sklearn()

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.base import is_classifier, is_regressor


# to reproduce errors even in CI
d4p.daalinit(nthreads=100)


def get_class_name(x):
    return x.__class__.__name__


WITHOUT_FIT = [
    'DBSCAN',
]


def func(X, Y, model, methods):
    clf = model
    if get_class_name(model) not in WITHOUT_FIT:
        clf.fit(X, Y)

    res = []
    name = []

    for i in methods:
        if i == 'predict':
            res.append(clf.predict(X))
            name.append(get_class_name(model) + '.predict(X)')
        elif i == 'predict_proba':
            res.append(clf.predict_proba(X))
            name.append(get_class_name(model) + '.predict_proba(X)')
        elif i == 'kneighbors':
            dist, idx = clf.kneighbors(X)
            res.append(dist)
            name.append('dist')
            res.append(idx)
            name.append('idx')
        elif i == 'fit_predict':
            predict = clf.fit_predict(X)
            res.append(predict)
            name.append(get_class_name(model) + '.fit_predict')
        elif i == 'transform':
            res.append(clf.transform(X))
            name.append(get_class_name(model) + '.transform(X)')
        elif i == 'get_covariance':
            res.append(clf.get_covariance())
            name.append(get_class_name(model) + '.get_covariance()')
        elif i == 'get_precision':
            res.append(clf.get_precision())
            name.append(get_class_name(model) + '.get_precision()')
        elif i == 'score_samples':
            res.append(clf.score_samples(X))
            name.append(get_class_name(model) + '.score_samples(X)')

    for i in clf.__dict__.keys():
        ans = getattr(clf, i)
        if isinstance(ans, (bool, float, int, np.ndarray, np.float64)):
            res.append(ans)
            name.append(get_class_name(model) + '.' + i)
    return res, name


def _run_test(model, methods):
    for features in [5, 10]:
        if get_class_name(model) in ['DBSCAN', 'KMeans']:
            X, y = make_blobs(n_samples=4000, n_features=features,
                              cluster_std=[1.0, 2.5, 0.5], random_state=0)
        elif is_classifier(model) or get_class_name(model) == 'PCA':
            X, y = make_classification(n_samples=4000, n_features=features,
                                       n_informative=features, n_redundant=0,
                                       n_clusters_per_class=8, random_state=0)
        elif is_regressor(model):
            X, y = make_regression(n_samples=4000, n_features=features,
                                   n_informative=features, random_state=0,
                                   noise=0.2, bias=10)
        else:
            raise ValueError('model must be classifier or regressor')

        baseline, name = func(X, y, model, methods)

        for i in range(10):
            res, _ = func(X, y, model, methods)

            for a, b, n in zip(res, baseline, name):
                np.testing.assert_allclose(a, b, rtol=0.0, atol=0.0,
                                           err_msg=str(n + " is incorrect"))


MODELS_INFO = [
    # ----------------------Passed----------------------
    {
        'model': KNeighborsClassifier(n_neighbors=10, algorithm='brute',
                                      weights="uniform"),
        'methods': ['predict', 'predict_proba', 'kneighbors'],
    },
    {
        'model': KNeighborsClassifier(n_neighbors=10, algorithm='brute',
                                      weights="distance"),
        'methods': ['predict', 'predict_proba', 'kneighbors'],
    },
    {
        'model': KNeighborsClassifier(n_neighbors=10, algorithm='kd_tree',
                                      weights="uniform"),
        'methods': ['predict', 'predict_proba', 'kneighbors'],
    },
    {
        'model': KNeighborsClassifier(n_neighbors=10, algorithm='kd_tree',
                                      weights="distance"),
        'methods': ['predict', 'predict_proba', 'kneighbors'],
    },
    {
        'model': DBSCAN(algorithm="brute", n_jobs=-1),
        'methods': ['fit_predict'],
    },
    {
        'model': SVC(random_state=0, probability=True, kernel='linear'),
        'methods': ['predict', 'predict_proba'],
    },
    {
        'model': SVC(random_state=0, probability=True, kernel='rbf'),
        'methods': ['predict', 'predict_proba'],
    },
    {
        'model': SVC(random_state=0, probability=True, kernel='rbf', gamma=0.01),
        'methods': ['predict', 'predict_proba'],
    },
    # ----------------------Failed----------------------
    {
        'model': KMeans(random_state=0, init="k-means++"),
        'methods': ['predict'],
    },
    {
        'model': KMeans(random_state=0, init="random"),
        'methods': ['predict'],
    },
    {
        'model': ElasticNet(random_state=0),
        'methods': ['predict'],
    },
    {
        'model': Lasso(random_state=0),
        'methods': ['predict'],
    },
    {
        'model': PCA(n_components=0.5, svd_solver="full", random_state=0),
        'methods': ['transform', 'get_covariance', 'get_precision', 'score_samples'],
    },
    # ----------------------Expected to be fixed in next release----------------------
    {
        'model': RandomForestClassifier(random_state=0, oob_score=True,
                                        max_samples=0.5, max_features='sqrt'),
        'methods': ['predict', 'predict_proba'],
    },
    {
        'model': LogisticRegression(random_state=0, solver="newton-cg", max_iter=1000),
        'methods': ['predict', 'predict_proba'],
    },
    {
        'model': LogisticRegression(random_state=0, solver="lbfgs", max_iter=1000),
        'methods': ['predict', 'predict_proba'],
    },
    {
        'model': LogisticRegressionCV(random_state=0, solver="newton-cg",
                                      n_jobs=-1, max_iter=1000),
        'methods': ['predict', 'predict_proba'],
    },
    {
        'model': LogisticRegressionCV(random_state=0, solver="lbfgs",
                                      n_jobs=-1, max_iter=1000),
        'methods': ['predict', 'predict_proba'],
    },
    {
        'model': RandomForestRegressor(random_state=0, oob_score=True,
                                       max_samples=0.5, max_features='sqrt'),
        'methods': ['predict'],
    },
    {
        'model': LinearRegression(),
        'methods': ['predict'],
    },
    {
        'model': Ridge(random_state=0),
        'methods': ['predict'],
    },
]

TO_SKIP = [
    'KMeans',
    'ElasticNet',
    'Lasso',
    'PCA',
    'RandomForestClassifier',
    'LogisticRegression',
    'LogisticRegressionCV',
    'RandomForestRegressor',
    'LinearRegression',
    'Ridge',
]


@pytest.mark.parametrize('model_head', MODELS_INFO)
def test_models(model_head):
    if get_class_name(model_head['model']) in TO_SKIP:
        pytest.skip("Unstable", allow_module_level=False)
    _run_test(model_head['model'], model_head['methods'])
