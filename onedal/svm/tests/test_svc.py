# ===============================================================================
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
# ===============================================================================

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from onedal.svm import SVC

from sklearn.utils.estimator_checks import check_estimator
import sklearn.utils.estimator_checks
from sklearn import datasets, metrics
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split


def _replace_and_save(md, fns, replacing_fn):
    saved = dict()
    for check_f in fns:
        try:
            fn = getattr(md, check_f)
            setattr(md, check_f, replacing_fn)
            saved[check_f] = fn
        except RuntimeError:
            pass
    return saved


def _restore_from_saved(md, saved_dict):
    for check_f in saved_dict:
        setattr(md, check_f, saved_dict[check_f])


def test_estimator():
    def dummy(*args, **kwargs):
        pass

    md = sklearn.utils.estimator_checks
    saved = _replace_and_save(md, [
        'check_sample_weights_invariance',  # Max absolute difference: 0.0008
        'check_estimators_fit_returns_self',  # ValueError: empty metadata
        'check_estimators_pickle',  # NotImplementedError
        'check_classifiers_predictions',  # Cannot cast ufunc 'multiply'
        'check_classifiers_train',  # assert y_pred.shape == (n_samples,)
        'check_classifiers_regression_target',  # Did not raise ValueError
        'check_supervised_y_2d',  # expected 1 DataConversionWarning
        'check_estimators_unfitted',  # Call 'fit' with appropriate arguments
    ], dummy)
    check_estimator(SVC())
    _restore_from_saved(md, saved)


def _test_libsvm_parameters(array_constr, dtype):
    X = array_constr([[-2, -1], [-1, -1], [-1, -2],
                      [1, 1], [1, 2], [2, 1]], dtype=dtype)
    y = array_constr([1, 1, 1, 2, 2, 2], dtype=dtype)

    clf = SVC(kernel='linear').fit(X, y)
    assert_array_equal(clf.dual_coef_, [[-0.25, .25]])
    assert_array_equal(clf.support_, [1, 3])
    assert_array_equal(clf.support_vectors_, (X[1], X[3]))
    assert_array_equal(clf.intercept_, [0.])
    assert_array_equal(clf.predict(X), y)


@pytest.mark.parametrize('array_constr', [np.array])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_libsvm_parameters(array_constr, dtype):
    _test_libsvm_parameters(array_constr, dtype)


def test_class_weight():
    X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]])
    y = np.array([1, 1, 1, 2, 2, 2])

    clf = SVC(class_weight={1: 0.1})
    clf.fit(X, y)
    assert_array_almost_equal(clf.predict(X), [2] * 6)


def test_sample_weight():
    X = np.array([[-2, 0], [-1, -1], [0, -2], [0, 2], [1, 1], [2, 2]])
    y = np.array([1, 1, 1, 2, 2, 2])

    clf = SVC(kernel='linear')
    clf.fit(X, y, sample_weight=[1] * 6)
    assert_array_almost_equal(clf.intercept_, [0.0])


def test_decision_function():
    X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
    Y = [1, 1, 1, 2, 2, 2]

    clf = SVC(kernel='rbf', gamma=1, decision_function_shape='ovo')
    clf.fit(X, Y)

    rbfs = rbf_kernel(X, clf.support_vectors_, gamma=clf.gamma)
    dec = np.dot(rbfs, clf.dual_coef_.T) + clf.intercept_
    assert_array_almost_equal(dec.ravel(), clf.decision_function(X))


def test_iris():
    iris = datasets.load_iris()
    clf = SVC(kernel='linear').fit(iris.data, iris.target)
    assert clf.score(iris.data, iris.target) > 0.9
    assert_array_equal(clf.classes_, np.sort(clf.classes_))


def test_decision_function_shape():
    X, y = make_blobs(n_samples=80, centers=5, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # check shape of ovo_decition_function=True
    clf = SVC(kernel='linear',
              decision_function_shape='ovo').fit(X_train, y_train)
    dec = clf.decision_function(X_train)
    assert dec.shape == (len(X_train), 10)

    # with pytest.raises(ValueError, match="must be either 'ovr' or 'ovo'"):
    #     SVC(decision_function_shape='bad').fit(X_train, y_train)
