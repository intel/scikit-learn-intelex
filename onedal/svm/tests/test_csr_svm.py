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
from scipy import sparse as sp

from numpy.testing import assert_array_equal, assert_array_almost_equal

from onedal.svm import SVC, SVR

from sklearn.utils.estimator_checks import check_estimator
import sklearn.utils.estimator_checks
from sklearn import datasets, metrics
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.base import clone as clone_estimator


def is_classifier(estimator):
    return getattr(estimator, "_estimator_type", None) == "classifier"


def check_svm_model_equal(svm, X_train, y_train, X_test):
    sparse_svm = clone_estimator(svm)
    dense_svm = clone_estimator(svm)
    dense_svm.fit(X_train.toarray(), y_train)
    if sp.isspmatrix(X_test):
        X_test_dense = X_test.toarray()
    else:
        X_test_dense = X_test
    sparse_svm.fit(X_train, y_train)
    assert sp.issparse(sparse_svm.support_vectors_)
    assert sp.issparse(sparse_svm.dual_coef_)
    assert_array_almost_equal(dense_svm.support_vectors_,
                              sparse_svm.support_vectors_.toarray())
    assert_array_almost_equal(dense_svm.dual_coef_,
                              sparse_svm.dual_coef_.toarray())
    assert_array_almost_equal(dense_svm.support_, sparse_svm.support_)
    assert_array_almost_equal(dense_svm.predict(X_test_dense),
                              sparse_svm.predict(X_test))

    if is_classifier(svm):
      assert_array_almost_equal(dense_svm.decision_function(X_test_dense),
                                sparse_svm.decision_function(X_test))
      assert_array_almost_equal(dense_svm.decision_function(X_test_dense),
                                sparse_svm.decision_function(X_test_dense))

def _test_binary_dataset(kernel):
    X, y = make_classification(n_samples=80, n_features=20, n_classes=2, random_state=0)
    sparse_X = sp.csr_matrix(X)

    dataset = sparse_X, y, sparse_X
    clf = SVC(kernel=kernel)
    check_svm_model_equal(clf, *dataset)


# @pytest.mark.parametrize('kernel', ['linear', 'rbf', 'poly'])
@pytest.mark.parametrize('kernel', ['linear', 'rbf'])
def test_binary_dataset(kernel):
    _test_binary_dataset(kernel)


def _test_iris(kernel):
    iris = datasets.load_iris()
    rng = np.random.RandomState(0)
    perm = rng.permutation(iris.target.size)
    iris.data = iris.data[perm]
    iris.target = iris.target[perm]
    sparse_iris_data = sp.csr_matrix(iris.data)

    dataset = sparse_iris_data, iris.target, sparse_iris_data

    clf = SVC(kernel=kernel)
    check_svm_model_equal(clf, *dataset)



# @pytest.mark.parametrize('kernel', ['linear', 'rbf', 'poly'])
@pytest.mark.parametrize('kernel', ['linear', 'rbf'])
def test_iris(kernel):
    _test_iris(kernel)

def _test_diabetes(kernel):
    diabetes = datasets.load_diabetes()

    sparse_diabetes_data = sp.csr_matrix(diabetes.data)
    dataset = sparse_diabetes_data, diabetes.target, sparse_diabetes_data

    clf = SVR(kernel=kernel, C=10.)
    check_svm_model_equal(clf, *dataset)


@pytest.mark.parametrize('kernel', ['linear', 'rbf'])
def test_diabetes(kernel):
    _test_diabetes(kernel)
