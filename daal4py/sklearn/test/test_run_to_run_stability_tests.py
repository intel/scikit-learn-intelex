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
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.base import is_classifier, is_regressor
from sklearn.metrics import pairwise_distances
from scipy import sparse


# to reproduce errors even in CI
d4p.daalinit(nthreads=20)


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
        elif i == 'fit_transform':
            res.append(clf.fit_transform(X))
            name.append(get_class_name(model) + '.fit_transform')
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


def _run_test(model, methods, dataset):
    for features in [5, 10]:
        if dataset == 'blobs':
            X, y = make_blobs(n_samples=4000, n_features=features,
                              cluster_std=[1.0, 2.5, 0.5], random_state=0)
        elif dataset in ['classifier', 'sparse']:
            X, y = make_classification(n_samples=4000, n_features=features,
                                       n_informative=features, n_redundant=0,
                                       n_clusters_per_class=8, random_state=0)
            if dataset == 'sparse':
                X = sparse.csr_matrix(X)
        elif dataset == 'regression':
            X, y = make_regression(n_samples=4000, n_features=features,
                                   n_informative=features, random_state=0,
                                   noise=0.2, bias=10)
        else:
            raise ValueError('Unknown dataset type')

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
        'dataset': 'classifier',
    },
    {
        'model': KNeighborsClassifier(n_neighbors=10, algorithm='brute',
                                      weights="distance"),
        'methods': ['predict', 'predict_proba', 'kneighbors'],
        'dataset': 'classifier',
    },
    {
        'model': KNeighborsClassifier(n_neighbors=10, algorithm='kd_tree',
                                      weights="uniform"),
        'methods': ['predict', 'predict_proba', 'kneighbors'],
        'dataset': 'classifier',
    },
    {
        'model': KNeighborsClassifier(n_neighbors=10, algorithm='kd_tree',
                                      weights="distance"),
        'methods': ['predict', 'predict_proba', 'kneighbors'],
        'dataset': 'classifier',
    },
    {
        'model': KNeighborsRegressor(n_neighbors=10, algorithm='kd_tree',
                                      weights="distance"),
        'methods': ['predict', 'kneighbors'],
        'dataset': 'regression',
    },
    {
        'model': KNeighborsRegressor(n_neighbors=10, algorithm='kd_tree',
                                      weights="uniform"),
        'methods': ['predict', 'kneighbors'],
        'dataset': 'regression',
    },
    {
        'model': KNeighborsRegressor(n_neighbors=10, algorithm='brute',
                                      weights="distance"),
        'methods': ['predict', 'kneighbors'],
        'dataset': 'regression',
    },
    {
        'model': KNeighborsRegressor(n_neighbors=10, algorithm='brute',
                                      weights="uniform"),
        'methods': ['predict', 'kneighbors'],
        'dataset': 'regression',
    },
    {
        'model': NearestNeighbors(n_neighbors=10, algorithm='brute'),
        'methods': ['kneighbors'],
        'dataset': 'blobs',
    },
    {
        'model': NearestNeighbors(n_neighbors=10, algorithm='kd_tree'),
        'methods': ['kneighbors'],
        'dataset': 'blobs',
    },
    {
        'model': TSNE(random_state=0),
        'methods': ['fit_transform'],
        'dataset': 'classifier',
    },
    {
        'model': DBSCAN(algorithm="brute", n_jobs=-1),
        'methods': ['fit_predict'],
        'dataset': 'blobs',
    },
    # {
    #     'model': SVC(random_state=0, probability=True, kernel='linear'),
    #     'methods': ['predict', 'predict_proba'],
    #     'dataset': 'classifier',
    # },
    {
        'model': SVC(random_state=0, probability=True, kernel='rbf'),
        'methods': ['predict', 'predict_proba'],
        'dataset': 'classifier',
    },
    # {
    #     'model': SVC(random_state=0, probability=True, kernel='linear'),
    #     'methods': ['predict', 'predict_proba'],
    #     'dataset': 'sparse',
    # },
    {
        'model': SVC(random_state=0, probability=True, kernel='rbf'),
        'methods': ['predict', 'predict_proba'],
        'dataset': 'sparse',
    },
    # ----------------------Failed----------------------
    {
        'model': KMeans(random_state=0, init="k-means++"),
        'methods': ['predict'],
        'dataset': 'blobs',
    },
    {
        'model': KMeans(random_state=0, init="random"),
        'methods': ['predict'],
        'dataset': 'blobs',
    },
    {
        'model': KMeans(random_state=0, init="k-means++"),
        'methods': ['sparse', 'predict'],
        'dataset': 'blobs',
    },
    {
        'model': KMeans(random_state=0, init="random"),
        'methods': ['sparse', 'predict'],
        'dataset': 'blobs',
    },
    {
        'model': ElasticNet(random_state=0),
        'methods': ['predict'],
        'dataset': 'regression',
    },
    {
        'model': Lasso(random_state=0),
        'methods': ['predict'],
        'dataset': 'regression',
    },
    {
        'model': PCA(n_components=0.5, svd_solver="full", random_state=0),
        'methods': ['transform', 'get_covariance', 'get_precision', 'score_samples'],
        'dataset': 'classifier',
    },
    # ----------------------Expected to be fixed in next release----------------------
    {
        'model': RandomForestClassifier(random_state=0, oob_score=True,
                                        max_samples=0.5, max_features='sqrt'),
        'methods': ['predict', 'predict_proba'],
        'dataset': 'classifier',
    },
    {
        'model': LogisticRegression(random_state=0, solver="newton-cg", max_iter=1000),
        'methods': ['predict', 'predict_proba'],
        'dataset': 'classifier',
    },
    {
        'model': LogisticRegression(random_state=0, solver="lbfgs", max_iter=1000),
        'methods': ['predict', 'predict_proba'],
        'dataset': 'classifier',
    },
    {
        'model': LogisticRegressionCV(random_state=0, solver="newton-cg",
                                      n_jobs=-1, max_iter=1000),
        'methods': ['predict', 'predict_proba'],
        'dataset': 'classifier',
    },
    {
        'model': LogisticRegressionCV(random_state=0, solver="lbfgs",
                                      n_jobs=-1, max_iter=1000),
        'methods': ['predict', 'predict_proba'],
        'dataset': 'classifier',
    },
    {
        'model': RandomForestRegressor(random_state=0, oob_score=True,
                                       max_samples=0.5, max_features='sqrt'),
        'methods': ['predict'],
        'dataset': 'regression',
    },
    {
        'model': LinearRegression(),
        'methods': ['predict'],
        'dataset': 'regression',
    },
    {
        'model': Ridge(random_state=0),
        'methods': ['predict'],
        'dataset': 'regression',
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
    _run_test(model_head['model'], model_head['methods'], model_head['dataset'])


@pytest.mark.parametrize('features', range(5, 10))
def test_train_test_split(features):
    X, y = make_classification(n_samples=4000, n_features=features,
                               n_informative=features, n_redundant=0,
                               n_clusters_per_class=8, random_state=0)
    baseline_X_train, baseline_X_test, baseline_y_train, baseline_y_test = \
        train_test_split(X, y, test_size=0.33, random_state=0)
    baseline = [baseline_X_train, baseline_X_test, baseline_y_train, baseline_y_test]
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, 
                                                            random_state=0)
        res = [X_train, X_test, y_train, y_test]
        for a, b in zip(res, baseline):
            np.testing.assert_allclose(a, b, rtol=0.0, atol=0.0,
                                       err_msg=str("train_test_split is incorrect"))


@pytest.mark.parametrize('metric', ['cosine', 'correlation'])
def test_pairwise_distances(metric):
    X = np.random.rand(1000)
    y = np.random.rand(1000)
    baseline = pairwise_distances(X.reshape(1, -1), y.reshape(1, -1), metric=metric)
    for i in range(5):
        res = pairwise_distances(X.reshape(1, -1), y.reshape(1, -1), metric=metric)
        for a, b in zip(res, baseline):
            np.testing.assert_allclose(a, b, rtol=0.0, atol=0.0,
                                       err_msg=str("pairwise_distances is incorrect"))
