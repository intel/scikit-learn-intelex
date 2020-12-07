#*******************************************************************************
# Copyright 2014-2020 Intel Corporation
# All Rights Reserved.
#
# This software is licensed under the Apache License, Version 2.0 (the
# "License"), the following terms apply:
#
# You may not use this file except in compliance with the License.  You may
# obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# See the License for the specific language governing permissions and
# limitations under the License.
#*******************************************************************************

import unittest
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

##################################################

def run_clf_test(func):
    for features in [5, 10]:
        X, y = make_classification(n_samples=4000, n_features=features, n_informative=features, n_redundant=0,
            n_clusters_per_class=8, random_state=0)

        baseline, name = func(X, y)

        for i in range(5):
            res, _ = func(X, y)

            for a, b, n in zip(res, baseline, name):
                np.testing.assert_allclose(a, b, rtol=0.0, atol=0.0, err_msg=str(n + " is incorrect"))

def run_reg_test(func):
    for features in [5, 10]:
        X, y = make_regression(n_samples=4000, n_features=features, n_informative=features, random_state=0, noise=0.2, bias=10)
        baseline, name = func(X, y)

        for i in range(5):
            res, _ = func(X, y)

            for a, b, n in zip(res, baseline, name):
                np.testing.assert_allclose(a, b, rtol=0.0, atol=0.0, err_msg=str(n + " is incorrect"))

##################################################

def run_rf_class(X, Y):
    clf = RandomForestClassifier(random_state=0, oob_score=True, max_samples=0.5, max_features='sqrt')
    clf.fit(X, Y)
    res  = [clf.predict(X), clf.predict_proba(X), clf.feature_importances_, clf.oob_score_]
    name = ["clf.predict(X)", "clf.predict_proba(X)", "clf.feature_importances_", "clf.oob_score_"]
    return res, name

def run_log_regression_newton(X, Y):
    clf = LogisticRegression(random_state=0, solver="newton-cg", max_iter=1000)
    clf.fit(X, Y)
    res  = [clf.predict(X), clf.predict_proba(X), clf.coef_, clf.intercept_, clf.n_iter_]
    name = ["clf.predict(X)", "clf.predict_proba(X)", "clf.coef_", "clf.intercept_", "clf.n_iter_"]
    return res, name

def run_log_regression_lbfgs(X, Y):
    clf = LogisticRegression(random_state=0, solver="lbfgs", max_iter=1000)
    clf.fit(X, Y)
    res  = [clf.predict(X), clf.predict_proba(X), clf.coef_, clf.intercept_, clf.n_iter_]
    name = ["clf.predict(X)", "clf.predict_proba(X)", "clf.coef_", "clf.intercept_", "clf.n_iter_"]
    return res, name

def run_log_regression_cv_newton(X, Y):
    clf = LogisticRegressionCV(random_state=0, solver="newton-cg", n_jobs=-1, max_iter=1000)
    clf.fit(X, Y)
    res  = [clf.predict(X), clf.predict_proba(X), clf.coef_, clf.intercept_, clf.n_iter_, clf.Cs_, clf.C_]
    name = ["clf.predict(X)", "clf.predict_proba(X)", "clf.coef_", "clf.intercept_", "clf.n_iter_", "clf.Cs_", "clf.C_"]
    return res, name

def run_log_regression_cv_lbfgs(X, Y):
    clf = LogisticRegressionCV(random_state=0, solver="lbfgs", n_jobs=-1, max_iter=1000)
    clf.fit(X, Y)
    res  = [clf.predict(X), clf.predict_proba(X), clf.coef_, clf.intercept_, clf.n_iter_, clf.Cs_, clf.C_]
    name = ["clf.predict(X)", "clf.predict_proba(X)", "clf.coef_", "clf.intercept_", "clf.n_iter_", "clf.Cs_", "clf.C_"]
    return res, name

def run_svc_linear(X, Y):
    clf = SVC(random_state=0, probability=True, kernel='linear')
    clf.fit(X, Y)
    res  = [clf.predict(X), clf.predict_proba(X), clf.support_, clf.support_vectors_, clf.n_support_, clf.dual_coef_, clf.coef_, clf.intercept_]
    name = ["clf.predict(X)", "clf.predict_proba(X)", "clf.support_", "clf.support_vectors_", "clf.n_support_", "clf.dual_coef_", "clf.coef_", "clf.intercept_"]
    return res, name

def run_svc_rbf(X, Y):
    clf = SVC(random_state=0, probability=True, kernel='rbf')
    clf.fit(X, Y)
    res  = [clf.predict(X), clf.predict_proba(X), clf.support_, clf.support_vectors_, clf.n_support_, clf.dual_coef_, clf.intercept_]
    name = ["clf.predict(X)", "clf.predict_proba(X)", "clf.support_", "clf.support_vectors_", "clf.n_support_", "clf.dual_coef_", "clf.coef_", "clf.intercept_"]
    return res, name

def run_svc_rbf_const_gamma(X, Y):
    clf = SVC(random_state=0, probability=True, kernel='rbf', gamma=0.01)
    clf.fit(X, Y)
    res  = [clf.predict(X), clf.predict_proba(X), clf.support_, clf.support_vectors_, clf.n_support_, clf.dual_coef_, clf.intercept_]
    name = ["clf.predict(X)", "clf.predict_proba(X)", "clf.support_", "clf.support_vectors_", "clf.n_support_", "clf.dual_coef_", "clf.coef_", "clf.intercept_"]
    return res, name

def run_knn_class_brute_uniform(X, Y):
    clf = KNeighborsClassifier(n_neighbors=10, algorithm='brute', weights="uniform")
    clf.fit(X, Y)

    dist, idx = clf.kneighbors(X)
    res  = [clf.predict(X), clf.predict_proba(X), dist, idx]
    name = ["clf.predict(X)", "clf.predict_proba(X)", "dist", "idx"]
    return res, name

def run_knn_class_brute_distance(X, Y):
    clf = KNeighborsClassifier(n_neighbors=10, algorithm='brute', weights="distance")
    clf.fit(X, Y)
    dist, idx = clf.kneighbors(X)
    res  = [clf.predict(X), clf.predict_proba(X), dist, idx]
    name = ["clf.predict(X)", "clf.predict_proba(X)", "dist", "idx"]
    return res, name

def run_knn_class_kdtree_uniform(X, Y):
    clf = KNeighborsClassifier(n_neighbors=10, algorithm='kd_tree', weights="uniform")
    clf.fit(X, Y)
    dist, idx = clf.kneighbors(X)
    res  = [clf.predict(X), clf.predict_proba(X), dist, idx]
    name = ["clf.predict(X)", "clf.predict_proba(X)", "dist", "idx"]
    return res, name

def run_knn_class_kdtree_distance(X, Y):
    clf = KNeighborsClassifier(n_neighbors=10, algorithm='kd_tree', weights="distance")
    clf.fit(X, Y)
    dist, idx = clf.kneighbors(X)
    res  = [clf.predict(X), clf.predict_proba(X), dist, idx]
    name = ["clf.predict(X)", "clf.predict_proba(X)", "dist", "idx"]
    return res, name

##################################################

def run_rf_regression(X, Y):
    clf = RandomForestRegressor(random_state=0, oob_score=True, max_samples=0.5, max_features='sqrt')
    clf.fit(X, Y)
    res  = [clf.predict(X),  clf.feature_importances_, clf.oob_score_, clf.oob_prediction_]
    name = ["clf.predict(X)",  "clf.feature_importances_", "clf.oob_score_", "clf.oob_prediction_"]
    return res, name

def run_linear_regression(X, Y):
    clf = LinearRegression()
    clf.fit(X, Y)
    res  = [clf.predict(X),  clf.coef_, clf.rank_, clf.singular_, clf.intercept_]
    name = ["clf.predict(X)", "clf.coef_", "clf.rank_", "clf.singular_", "clf.intercept_"]
    return res, name

def run_ridge_regression(X, Y):
    clf = Ridge(random_state=0)
    clf.fit(X, Y)
    res  = [clf.predict(X),  clf.coef_, clf.n_iter_, clf.intercept_]
    name = ["clf.predict(X)", "clf.coef_", "clf.n_iter_", "clf.intercept_"]
    return res, name

def run_elasticnet_regression(X, Y):
    clf = ElasticNet(random_state=0)
    clf.fit(X, Y)
    res  = [clf.predict(X),  clf.coef_, clf.n_iter_, clf.intercept_]
    name = ["clf.predict(X)", "clf.coef_", "clf.n_iter_", "clf.intercept_"]
    return res, name

def run_lasso_regression(X, Y):
    clf = Lasso(random_state=0)
    clf.fit(X, Y)
    res  = [clf.predict(X),  clf.coef_, clf.n_iter_, clf.intercept_]
    name = ["clf.predict(X)", "clf.coef_", "clf.n_iter_", "clf.intercept_"]
    return res, name

##################################################

def run_kmeans_plus_plus(X, Y):
    clf = KMeans(random_state=0, init="k-means++")
    clf.fit(X, Y)
    res  = [clf.predict(X),  clf.cluster_centers_, clf.labels_, clf.inertia_, clf.n_iter_]
    name = ["clf.predict(X)",  "clf.cluster_centers_", "clf.labels_", "clf.inertia_", "clf.n_iter_"]
    return res, name

def run_kmeans_random(X, Y):
    clf = KMeans(random_state=0, init="random")
    clf.fit(X, Y)
    res  = [clf.predict(X),  clf.cluster_centers_, clf.labels_, clf.inertia_, clf.n_iter_]
    name = ["clf.predict(X)",  "clf.cluster_centers_", "clf.labels_", "clf.inertia_", "clf.n_iter_"]
    return res, name

def run_dbascan(X, Y):
    clf = DBSCAN(algorithm="brute", n_jobs=-1)
    predict = clf.fit_predict(X)
    res  = [predict, clf.core_sample_indices_, clf.components_, clf.labels_]
    name = ["predict", "clf.core_sample_indices_", "clf.components_", "clf.labels_"]
    return res, name

##################################################

def run_pca_full(X, Y):
    clf = PCA(n_components=0.5, svd_solver="full", random_state=0)
    clf.fit(X)
    res  = [clf.transform(X), clf.get_covariance(), clf.get_precision(), clf.score_samples(X), clf.components_, clf.explained_variance_, clf.explained_variance_ratio_, clf.singular_values_, clf.mean_, clf.noise_variance_]
    name = ["clf.transform(X)", "clf.get_covariance()", "clf.get_precision()", "clf.score_samples(X)", "clf.components_", "clf.explained_variance_", "clf.explained_variance_ratio_", "clf.singular_values_", "clf.mean_", "clf.noise_variance_"]
    return res, name

def run_pca_daal4py_corr(X, Y):
    algo = d4p.pca(resultsToCompute="mean|variance|eigenvalue", isDeterministic=True, method="correlationDense")
    result1 = algo.compute(X)

    pcatrans_algo = d4p.pca_transform(nComponents=X.shape[1]//2)
    transform = pcatrans_algo.compute(X, result1.eigenvectors, result1.dataForTransform).transformedData

    res  = [transform, result1.eigenvalues, result1.eigenvectors, result1.means, result1.variances]
    name = ["transform", "result1.eigenvalues", "result1.eigenvectors", "result1.means", "result1.variances"]
    return res, name

##################################################

class Test(unittest.TestCase):

    #---------------------Passed---------------------

    def test_knn_class_brute_uniform(self):
        run_clf_test(run_knn_class_brute_uniform)

    def test_knn_class_brute_distance(self):
        run_clf_test(run_knn_class_brute_distance)

    def test_knn_class_kdtree_uniform(self):
        run_clf_test(run_knn_class_kdtree_uniform)

    def test_knn_class_kdtree_distance(self):
        run_clf_test(run_knn_class_kdtree_distance)

    def test_dbascan(self):
        run_clf_test(run_dbascan) 

    def test_svc_linear(self):
        run_clf_test(run_svc_linear)

    def test_svc_rbf(self):
        run_clf_test(run_svc_rbf)

    def test_svc_rbf_const_gamma(self):
        run_clf_test(run_svc_rbf_const_gamma)

    #---------------------Failed---------------------

    @unittest.expectedFailure
    def test_kmeans_plus_plus(self):
        run_clf_test(run_kmeans_plus_plus)

    @unittest.expectedFailure
    def test_kmeans_random(self):
        run_clf_test(run_kmeans_random)

    @unittest.expectedFailure
    def test_elasticnet_regression(self):
        run_reg_test(run_elasticnet_regression)

    @unittest.expectedFailure
    def test_lasso_regression(self):
        run_reg_test(run_lasso_regression)

    @unittest.expectedFailure
    def test_pca_full(self):
        run_clf_test(run_pca_full)

    @unittest.expectedFailure
    def test_pca_daal4py_corr(self):
        run_clf_test(run_pca_daal4py_corr)

    #---------------------Expected to be fixed in next release---------------------

    @unittest.expectedFailure
    def test_log_regression_newton(self):
        run_clf_test(run_log_regression_newton)

    @unittest.expectedFailure
    def test_log_regression_lbfgs(self):
        run_clf_test(run_log_regression_lbfgs)

    @unittest.expectedFailure
    def test_log_regression_cv_newton(self):
        run_clf_test(run_log_regression_cv_newton)

    @unittest.expectedFailure
    def test_log_regression_cv_lbfgs(self):
        run_clf_test(run_log_regression_cv_lbfgs)

    @unittest.expectedFailure
    def test_linear_regression(self):
        run_reg_test(run_linear_regression)

    @unittest.expectedFailure
    def test_ridge_regression(self):
        run_reg_test(run_ridge_regression)

    @unittest.expectedFailure
    def test_rf_class(self):
        run_clf_test(run_rf_class)

    @unittest.expectedFailure
    def test_rf_reg(self):
        run_reg_test(run_rf_regression)


if __name__ == '__main__':
    unittest.main()
