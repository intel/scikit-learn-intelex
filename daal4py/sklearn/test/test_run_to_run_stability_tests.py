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
import daal4py as d4p

from daal4py.sklearn import patch_sklearn
patch_sklearn()

# to reproduce errors even in CI
d4p.daalinit(nthreads=100)

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression

import pytest

WITHOUT_FIT = [
    'DBSCAN',
    'PCA_daal4py_corr',
]

TO_COMPUTE = [
    'PCA_daal4py_corr',
]

def func(X, Y, model_name, model, attribute, methods):
    clf = model
    if model_name not in WITHOUT_FIT:
        clf.fit(X, Y)
    if model_name in TO_COMPUTE:
        clf = clf.compute(X)

    res = []
    name = []
    
    if model_name == 'PCA_daal4py_corr':
        pcatrans_algo = d4p.pca_transform(nComponents=X.shape[1]//2)
        transform = pcatrans_algo.compute(X, clf.eigenvectors, clf.dataForTransform).transformedData
        res.append(transform)
        name.append('transform')

    for i in methods:
        if i == 'predict':
            res.append(clf.predict(X))
            name.append('clf.predict(X)')
        elif i == 'predict_proba':
            res.append(clf.predict_proba(X))
            name.append('clf.predict_proba(X)')
        elif i == 'kneighbors':
            dist, idx = clf.kneighbors(X)
            res.append(dist)
            name.append('dist')
            res.append(idx)
            name.append('idx')
        elif i == 'fit_predict':
            predict = clf.fit_predict(X)
            res.append(predict)
            name.append('predict')
        elif i == 'transform':
            res.append(clf.transform(X))
            name.append('clf.transform(X)')
        elif i == 'get_covariance':
            res.append(clf.get_covariance())
            name.append('clf.get_covariance()')
        elif i == 'get_precision':
            res.append(clf.get_precision())
            name.append('clf.get_precision()')
        elif i == 'score_samples':
            res.append(clf.score_samples(X))
            name.append('clf.score_samples(X)')

    for i in attribute:
        res.append(getattr(clf, i))
        name.append('clf.' + i)
    return res, name
    

def _run_test(model_type, model_name, model, attribute, methods):
    for features in [5, 10]:
        if model_type == 'clf':
            X, y = make_classification(n_samples=4000, n_features=features, 
                                       n_informative=features, n_redundant=0,
                                       n_clusters_per_class=8, random_state=0)
        elif model_type == 'reg':
            X, y = make_regression(n_samples=4000, n_features=features, 
                                   n_informative=features, random_state=0, 
                                   noise=0.2, bias=10)

        baseline, name = func(X, y, model_name, model, attribute, methods)

        for i in range(10):
            res, _ = func(X, y, model_name, model, attribute, methods)

            for a, b, n in zip(res, baseline, name):
                np.testing.assert_allclose(a, b, rtol=0.0, atol=0.0, err_msg=str(n + " is incorrect"))


MODELS_INFO = [
    #----------------------Passed----------------------
    {
        'model_type'    : 'clf', 
        'model_name'    : 'KNN_brute_uniform', 
        'model'         : KNeighborsClassifier(n_neighbors=10, algorithm='brute', weights="uniform"),
        'attributes'    : [],
        'methods'       : ['predict', 'predict_proba', 'kneighbors'],
    },
    {
        'model_type'    : 'clf', 
        'model_name'    : 'KNN_brute_distance', 
        'model'         : KNeighborsClassifier(n_neighbors=10, algorithm='brute', weights="distance"),
        'attributes'    : [],
        'methods'       : ['predict', 'predict_proba', 'kneighbors'],
    },
    {
        'model_type'    : 'clf', 
        'model_name'    : 'KNN_kd_tree_uniform', 
        'model'         : KNeighborsClassifier(n_neighbors=10, algorithm='kd_tree', weights="uniform"),
        'attributes'    : [],
        'methods'       : ['predict', 'predict_proba', 'kneighbors'],
    },
    {
        'model_type'    : 'clf', 
        'model_name'    : 'KNN_kd_tree_distance', 
        'model'         : KNeighborsClassifier(n_neighbors=10, algorithm='kd_tree', weights="distance"),
        'attributes'    : [],
        'methods'       : ['predict', 'predict_proba', 'kneighbors'],
    },
    {
        'model_type'    : 'clf', 
        'model_name'    : 'DBSCAN', 
        'model'         : DBSCAN(algorithm="brute", n_jobs=-1),
        'attributes'    : ['core_sample_indices_', 'components_', 'labels_'],
        'methods'       : ['fit_predict'],
    },
    {
        'model_type'    : 'clf', 
        'model_name'    : 'SVC_linear', 
        'model'         : SVC(random_state=0, probability=True, kernel='linear'),
        'attributes'    : ['support_', 'support_vectors_', 'n_support_', 'dual_coef_', 'coef_', 'intercept_'],
        'methods'       : ['predict', 'predict_proba'],
    },
    {
        'model_type'    : 'clf', 
        'model_name'    : 'SVC_rbf', 
        'model'         : SVC(random_state=0, probability=True, kernel='rbf'),
        'attributes'    : ['support_', 'support_vectors_', 'n_support_', 'dual_coef_', 'intercept_'],
        'methods'       : ['predict', 'predict_proba'],
    },
    {
        'model_type'    : 'clf', 
        'model_name'    : 'SVC_rbf_const_gamma', 
        'model'         : SVC(random_state=0, probability=True, kernel='rbf', gamma=0.01),
        'attributes'    : ['support_', 'support_vectors_', 'n_support_', 'dual_coef_', 'intercept_'],
        'methods'       : ['predict', 'predict_proba'],
    },
    #----------------------Failed----------------------
    {
        'model_type'    : 'clf', 
        'model_name'    : 'KMeans_plus_plus', 
        'model'         : KMeans(random_state=0, init="k-means++"),
        'attributes'    : ['cluster_centers_', 'labels_', 'inertia_', 'n_iter_'],
        'methods'       : ['predict'],
    },
    {
        'model_type'    : 'clf', 
        'model_name'    : 'KMeans_random', 
        'model'         : KMeans(random_state=0, init="random"),
        'attributes'    : ['cluster_centers_', 'labels_', 'inertia_', 'n_iter_'],
        'methods'       : ['predict'],
    },
    {
        'model_type'    : 'reg', 
        'model_name'    : 'ElasticNet', 
        'model'         : ElasticNet(random_state=0),
        'attributes'    : ['coef_', 'n_iter_', 'intercept_'],
        'methods'       : ['predict'],
    },
    {
        'model_type'    : 'reg', 
        'model_name'    : 'Lasso', 
        'model'         : Lasso(random_state=0),
        'attributes'    : ['coef_', 'n_iter_', 'intercept_'],
        'methods'       : ['predict'],
    },
    {
        'model_type'    : 'clf', 
        'model_name'    : 'PCA_full', 
        'model'         : PCA(n_components=0.5, svd_solver="full", random_state=0),
        'attributes'    : ['components_', 'explained_variance_', 'explained_variance_ratio_', 'singular_values_', 'mean_', 'noise_variance_'],
        'methods'       : ['transform', 'get_covariance', 'get_precision', 'score_samples'],
    },
    {
        'model_type'    : 'clf', 
        'model_name'    : 'PCA_daal4py_corr', 
        'model'         : d4p.pca(resultsToCompute="mean|variance|eigenvalue", isDeterministic=True, method="correlationDense"),
        'attributes'    : ['eigenvalues', 'eigenvectors', 'means', 'variances'],
        'methods'       : [],
    },
    #----------------------Expected to be fixed in next release----------------------
    {
        'model_type'    : 'clf', 
        'model_name'    : 'RandomForestClassifier', 
        'model'         : RandomForestClassifier(random_state=0, oob_score=True, max_samples=0.5, max_features='sqrt'),
        'attributes'    : ['feature_importances_', 'oob_score_'],
        'methods'       : ['predict', 'predict_proba'],
    },
    {
        'model_type'    : 'clf', 
        'model_name'    : 'LogisticRegression_newton', 
        'model'         : LogisticRegression(random_state=0, solver="newton-cg", max_iter=1000),
        'attributes'    : ['coef_', 'intercept_', 'n_iter_'],
        'methods'       : ['predict', 'predict_proba'],
    },
    {
        'model_type'    : 'clf', 
        'model_name'    : 'LogisticRegression_lbfgs', 
        'model'         : LogisticRegression(random_state=0, solver="lbfgs", max_iter=1000),
        'attributes'    : ['coef_', 'intercept_', 'n_iter_'],
        'methods'       : ['predict', 'predict_proba'],
    },
    {
        'model_type'    : 'clf', 
        'model_name'    : 'LogisticRegressionCV_newton', 
        'model'         : LogisticRegressionCV(random_state=0, solver="newton-cg", n_jobs=-1, max_iter=1000),
        'attributes'    : ['coef_', 'intercept_', 'n_iter_', 'Cs_', 'C_'],
        'methods'       : ['predict', 'predict_proba'],
    },
    {
        'model_type'    : 'clf', 
        'model_name'    : 'LogisticRegressionCV_lbfgs', 
        'model'         : LogisticRegressionCV(random_state=0, solver="lbfgs", n_jobs=-1, max_iter=1000),
        'attributes'    : ['coef_', 'intercept_', 'n_iter_', 'Cs_', 'C_'],
        'methods'       : ['predict', 'predict_proba'],
    },
    {
        'model_type'    : 'reg', 
        'model_name'    : 'RandomForestRegressor', 
        'model'         : RandomForestRegressor(random_state=0, oob_score=True, max_samples=0.5, max_features='sqrt'),
        'attributes'    : ['feature_importances_', 'oob_score_', 'oob_prediction_'],
        'methods'       : ['predict'],
    },
    {
        'model_type'    : 'reg', 
        'model_name'    : 'LinearRegression', 
        'model'         : LinearRegression(),
        'attributes'    : ['coef_', 'rank_', 'singular_', 'intercept_'],
        'methods'       : ['predict'],
    },
    {
        'model_type'    : 'reg', 
        'model_name'    : 'Ridge', 
        'model'         : Ridge(random_state=0),
        'attributes'    : ['coef_', 'n_iter_', 'intercept_'],
        'methods'       : ['predict'],
    }, 
]

TO_SKIP = [
    'KMeans_plus_plus', 
    'KMeans_random', 
    'ElasticNet', 
    'Lasso', 
    'PCA_full', 
    'PCA_daal4py_corr', 
    'RandomForestClassifier', 
    'LogisticRegression_newton', 
    'LogisticRegression_lbfgs', 
    'LogisticRegressionCV_newton', 
    'LogisticRegressionCV_lbfgs', 
    'RandomForestRegressor', 
    'LinearRegression', 
    'Ridge', 
]


@pytest.mark.parametrize('model_head', MODELS_INFO)
def test_models(model_head):
    if model_head['model_name'] in TO_SKIP:
        pytest.skip("UNSTABLE", allow_module_level=False)
    _run_test(model_head['model_type'], model_head['model_name'], model_head['model'], model_head['attributes'], model_head['methods'])
