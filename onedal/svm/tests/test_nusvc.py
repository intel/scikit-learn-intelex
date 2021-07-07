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

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from onedal.svm import NuSVC
from sklearn.svm import NuSVC as SklearnNuSVC

from sklearn import datasets
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split


def _test_libsvm_parameters(array_constr, dtype):
    X = array_constr([[-2, -1], [-1, -1], [-1, -2],
                      [1, 1], [1, 2], [2, 1]], dtype=dtype)
    y = array_constr([1, 1, 1, 2, 2, 2], dtype=dtype)

    clf = NuSVC(kernel='linear').fit(X, y)
    assert_array_almost_equal(
        clf.dual_coef_, [[-0.04761905, -0.0952381, 0.0952381, 0.04761905]])
    assert_array_equal(clf.support_, [0, 1, 3, 4])
    assert_array_equal(clf.support_vectors_, X[clf.support_])
    assert_array_equal(clf.intercept_, [0.])
    assert_array_equal(clf.predict(X), y)


@pytest.mark.parametrize('array_constr', [np.array])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_libsvm_parameters(array_constr, dtype):
    _test_libsvm_parameters(array_constr, dtype)


def test_class_weight():
    X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]])
    y = np.array([1, 1, 1, 2, 2, 2])

    clf = NuSVC(class_weight={1: 0.1})
    clf.fit(X, y)
    assert_array_almost_equal(clf.predict(X), [2] * 6)


def test_sample_weight():
    X = np.array([[-2, 0], [-1, -1], [0, -2], [0, 2], [1, 1], [2, 2]])
    y = np.array([1, 1, 1, 2, 2, 2])

    clf = NuSVC(kernel='linear')
    clf.fit(X, y, sample_weight=[1] * 6)
    assert_array_almost_equal(clf.intercept_, [0.0])


def test_decision_function():
    X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
    Y = [1, 1, 1, 2, 2, 2]

    clf = NuSVC(kernel='rbf', gamma=1, decision_function_shape='ovo')
    clf.fit(X, Y)

    rbfs = rbf_kernel(X, clf.support_vectors_, gamma=clf.gamma)
    dec = np.dot(rbfs, clf.dual_coef_.T) + clf.intercept_
    assert_array_almost_equal(dec.ravel(), clf.decision_function(X))


def test_iris():
    iris = datasets.load_iris()
    clf = NuSVC(kernel='linear').fit(iris.data, iris.target)
    assert clf.score(iris.data, iris.target) > 0.9
    assert_array_equal(clf.classes_, np.sort(clf.classes_))


def test_decision_function_shape():
    X, y = make_blobs(n_samples=80, centers=5, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # check shape of ovo_decition_function=True
    clf = NuSVC(kernel='linear',
                decision_function_shape='ovo').fit(X_train, y_train)
    dec = clf.decision_function(X_train)
    assert dec.shape == (len(X_train), 10)

    # with pytest.raises(ValueError, match="must be either 'ovr' or 'ovo'"):
    #     SVC(decision_function_shape='bad').fit(X_train, y_train)


def test_pickle():
    iris = datasets.load_iris()
    clf = NuSVC(kernel='linear').fit(iris.data, iris.target)
    expected = clf.decision_function(iris.data)

    import pickle
    dump = pickle.dumps(clf)
    clf2 = pickle.loads(dump)

    assert type(clf2) == clf.__class__
    result = clf2.decision_function(iris.data)
    assert_array_equal(expected, result)


def _test_cancer_rbf_compare_with_sklearn(nu, gamma):
    cancer = datasets.load_breast_cancer()

    clf = NuSVC(kernel='rbf', gamma=gamma, nu=nu)
    clf.fit(cancer.data, cancer.target)
    result = clf.score(cancer.data, cancer.target)

    clf = SklearnNuSVC(kernel='rbf', gamma=gamma, nu=nu)
    clf.fit(cancer.data, cancer.target)
    expected = clf.score(cancer.data, cancer.target)

    assert result > 0.4
    assert abs(result - expected) < 1e-4


@pytest.mark.parametrize('gamma', ['scale', 'auto'])
@pytest.mark.parametrize('nu', [0.25, 0.5])
def test_cancer_rbf_compare_with_sklearn(nu, gamma):
    _test_cancer_rbf_compare_with_sklearn(nu, gamma)


def _test_cancer_linear_compare_with_sklearn(nu):
    cancer = datasets.load_breast_cancer()

    clf = NuSVC(kernel='linear', nu=nu)
    clf.fit(cancer.data, cancer.target)
    result = clf.score(cancer.data, cancer.target)

    clf = SklearnNuSVC(kernel='linear', nu=nu)
    clf.fit(cancer.data, cancer.target)
    expected = clf.score(cancer.data, cancer.target)

    assert result > 0.5
    assert abs(result - expected) < 1e-3


@pytest.mark.parametrize('nu', [0.25, 0.5])
def test_cancer_linear_compare_with_sklearn(nu):
    _test_cancer_linear_compare_with_sklearn(nu)


def _test_cancer_poly_compare_with_sklearn(params):
    cancer = datasets.load_breast_cancer()

    clf = NuSVC(kernel='poly', **params)
    clf.fit(cancer.data, cancer.target)
    result = clf.score(cancer.data, cancer.target)

    clf = SklearnNuSVC(kernel='poly', **params)
    clf.fit(cancer.data, cancer.target)
    expected = clf.score(cancer.data, cancer.target)

    assert result > 0.5
    assert abs(result - expected) < 1e-4


@pytest.mark.parametrize('params', [
    {'degree': 2, 'coef0': 0.1, 'gamma': 'scale', 'nu': .25},
    {'degree': 3, 'coef0': 0.0, 'gamma': 'scale', 'nu': .5}
])
def test_cancer_poly_compare_with_sklearn(params):
    _test_cancer_poly_compare_with_sklearn(params)
