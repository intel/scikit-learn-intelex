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

from onedal.svm import SVC

from sklearn.utils.estimator_checks import check_estimator
import sklearn.utils.estimator_checks
from sklearn import datasets, metrics
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split


def check_svm_model_equal(dense_svm, sparse_svm, X_train, y_train, X_test):
    dense_svm.fit(X_train.toarray(), y_train)
    if sp.isspmatrix(X_test):
        X_test_dense = X_test.toarray()
    else:
        X_test_dense = X_test
    sparse_svm.fit(X_train, y_train)
    assert sp.issparse(sparse_svm.support_vectors_)
    # assert sp.issparse(sparse_svm.dual_coef_)
    assert_array_almost_equal(dense_svm.support_vectors_,
                              sparse_svm.support_vectors_.toarray())
    assert_array_almost_equal(dense_svm.dual_coef_,
                              sparse_svm.dual_coef_.toarray())
    assert_array_almost_equal(dense_svm.support_, sparse_svm.support_)
    assert_array_almost_equal(dense_svm.predict(X_test_dense),
                              sparse_svm.predict(X_test))
    assert_array_almost_equal(dense_svm.decision_function(X_test_dense),
                              sparse_svm.decision_function(X_test))
    assert_array_almost_equal(dense_svm.decision_function(X_test_dense),
                              sparse_svm.decision_function(X_test_dense))

    assert_array_almost_equal(dense_svm.predict_proba(X_test_dense),
                              sparse_svm.predict_proba(X_test), 4)
    msg = "cannot use sparse input in 'SVC' trained on dense data"
    if sp.isspmatrix(X_test):
        assert_raise_message(ValueError, msg, dense_svm.predict, X_test)


# @pytest.mark.parametrize('kernel', ['linear', 'rbf', 'poly'])
@pytest.mark.parametrize('kernel', ['linear'])
def test_iris(kernel):
    _test_iris(kernel)


def _test_iris(kernel):
    iris = datasets.load_iris()
    rng = np.random.RandomState(0)
    perm = rng.permutation(iris.target.size)
    iris.data = iris.data[perm]
    iris.target = iris.target[perm]
    iris.data = sp.csr_matrix(iris.data)

    dataset = iris.data, iris.target, iris.data

    clf = SVC(kernel=kernel, probability=True,
                  random_state=0, decision_function_shape='ovo')
    sp_clf = SVC(kernel=kernel, probability=True,
                     random_state=0, decision_function_shape='ovo')
    check_svm_model_equal(clf, sp_clf, *dataset)



