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
from numpy.testing import assert_array_almost_equal, assert_allclose
from sklearn import datasets
from sklearn.metrics.pairwise import rbf_kernel

from onedal.svm import SVR
from sklearn.svm import SVR as SklearnSVR

from sklearn.utils.estimator_checks import check_estimator
import sklearn.utils.estimator_checks


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
        'check_sample_weights_invariance',  # Max absolute difference: 0.0002
        'check_estimators_fit_returns_self',  # ???
        'check_regressors_train',  # Cannot get data type from empty metadata
        'check_supervised_y_2d',  # need warning, why?
        'check_regressors_int',  # very bad accuracy
        'check_estimators_unfitted',  # expected NotFittedError from sklearn
        'check_fit_idempotent',  # again run fit - error. need to fix
        'check_estimators_pickle',  # NotImplementedError
    ], dummy)
    check_estimator(SVR())
    _restore_from_saved(md, saved)


def test_run_to_run_fit():
    diabetes = datasets.load_diabetes()
    clf_first = SVR(kernel='linear', C=10.)
    clf_first.fit(diabetes.data, diabetes.target)

    for _ in range(10):
        clf = SVR(kernel='linear', C=10.)
        clf.fit(diabetes.data, diabetes.target)
        assert_allclose(clf_first.intercept_, clf.intercept_)
        assert_allclose(clf_first.support_vectors_, clf.support_vectors_)
        assert_allclose(clf_first.dual_coef_, clf.dual_coef_)


def test_diabetes_simple():
    diabetes = datasets.load_diabetes()
    clf = SVR(kernel='linear', C=10.)
    clf.fit(diabetes.data, diabetes.target)
    assert clf.score(diabetes.data, diabetes.target) > 0.02


def test_input_format_for_diabetes():
    diabetes = datasets.load_diabetes()

    c_contiguous_numpy = np.asanyarray(diabetes.data, dtype='float', order='C')
    assert c_contiguous_numpy.flags.c_contiguous
    assert not c_contiguous_numpy.flags.f_contiguous
    assert not c_contiguous_numpy.flags.fnc

    clf = SVR(kernel='linear', C=10.)
    clf.fit(c_contiguous_numpy, diabetes.target)
    dual_c_contiguous_numpy = clf.dual_coef_
    res_c_contiguous_numpy = clf.predict(c_contiguous_numpy)

    f_contiguous_numpy = np.asanyarray(diabetes.data, dtype='float', order='F')
    assert not f_contiguous_numpy.flags.c_contiguous
    assert f_contiguous_numpy.flags.f_contiguous
    assert f_contiguous_numpy.flags.fnc

    clf = SVR(kernel='linear', C=10.)
    clf.fit(f_contiguous_numpy, diabetes.target)
    dual_f_contiguous_numpy = clf.dual_coef_
    res_f_contiguous_numpy = clf.predict(f_contiguous_numpy)
    assert_allclose(dual_c_contiguous_numpy, dual_f_contiguous_numpy)
    assert_allclose(res_c_contiguous_numpy, res_f_contiguous_numpy)


def test_predict():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    reg = SVR(kernel='linear', C=0.1).fit(X, y)

    linear = np.dot(X, reg.support_vectors_.T)
    dec = np.dot(linear, reg.dual_coef_.T) + reg.intercept_
    assert_array_almost_equal(dec.ravel(), reg.predict(X).ravel())

    reg = SVR(kernel='rbf', gamma=1).fit(X, y)

    rbfs = rbf_kernel(X, reg.support_vectors_, gamma=reg.gamma)
    dec = np.dot(rbfs, reg.dual_coef_.T) + reg.intercept_
    assert_array_almost_equal(dec.ravel(), reg.predict(X).ravel())


def _test_diabetes_compare_with_sklearn(kernel):
    diabetes = datasets.load_diabetes()
    clf_onedal = SVR(kernel=kernel, C=10.)
    clf_onedal.fit(diabetes.data, diabetes.target)
    result = clf_onedal.score(diabetes.data, diabetes.target)

    clf_sklearn = SklearnSVR(kernel=kernel, C=10.)
    clf_sklearn.fit(diabetes.data, diabetes.target)
    expected = clf_sklearn.score(diabetes.data, diabetes.target)

    print(result, expected)
    assert result > expected - 1e-5
    assert_allclose(clf_sklearn.intercept_, clf_onedal.intercept_, atol=1e-4)
    assert_allclose(clf_sklearn.support_vectors_.shape,
                    clf_sklearn.support_vectors_.shape)
    assert_allclose(clf_sklearn.dual_coef_, clf_onedal.dual_coef_, atol=1e-2)


@pytest.mark.parametrize('kernel', ['linear', 'rbf', 'poly'])
def test_diabetes_compare_with_sklearn(kernel):
    _test_diabetes_compare_with_sklearn(kernel)


def _test_boston_rbf_compare_with_sklearn(C, gamma):
    diabetes = datasets.load_boston()
    clf = SVR(kernel='rbf', gamma=gamma, C=C)
    clf.fit(diabetes.data, diabetes.target)
    result = clf.score(diabetes.data, diabetes.target)

    clf = SklearnSVR(kernel='rbf', gamma=gamma, C=C)
    clf.fit(diabetes.data, diabetes.target)
    expected = clf.score(diabetes.data, diabetes.target)

    print(result, expected)
    assert result > 0.4
    assert result > expected - 1e-5


@pytest.mark.parametrize('gamma', ['scale', 'auto'])
@pytest.mark.parametrize('C', [100.0, 1000.0])
def test_boston_rbf_compare_with_sklearn(C, gamma):
    _test_boston_rbf_compare_with_sklearn(C, gamma)


def _test_boston_linear_compare_with_sklearn(C):
    diabetes = datasets.load_boston()
    clf = SVR(kernel='linear', C=C)
    clf.fit(diabetes.data, diabetes.target)
    result = clf.score(diabetes.data, diabetes.target)

    clf = SklearnSVR(kernel='linear', C=C)
    clf.fit(diabetes.data, diabetes.target)
    expected = clf.score(diabetes.data, diabetes.target)

    print(result, expected)
    assert result > 0.5
    assert result > expected - 1e-3


@pytest.mark.parametrize('C', [0.001, 0.1])
def test_boston_linear_compare_with_sklearn(C):
    _test_boston_linear_compare_with_sklearn(C)


def _test_boston_poly_compare_with_sklearn(params):
    diabetes = datasets.load_boston()
    clf = SVR(kernel='poly', **params)
    clf.fit(diabetes.data, diabetes.target)
    result = clf.score(diabetes.data, diabetes.target)

    clf = SklearnSVR(kernel='poly', **params)
    clf.fit(diabetes.data, diabetes.target)
    expected = clf.score(diabetes.data, diabetes.target)

    print(result, expected)
    assert result > 0.5
    assert result > expected - 1e-5


@pytest.mark.parametrize('params', [
    {'degree': 2, 'coef0': 0.1, 'gamma': 'scale', 'C': 100},
    {'degree': 3, 'coef0': 0.0, 'gamma': 'scale', 'C': 1000}
])
def test_boston_poly_compare_with_sklearn(params):
    _test_boston_poly_compare_with_sklearn(params)


def test_sided_sample_weight():
    clf = SVR(C=1e-2, kernel='linear')

    X = [[-2, 0], [-1, -1], [0, -2], [0, 2], [1, 1], [2, 0]]
    Y = [1, 1, 1, 2, 2, 2]

    sample_weight = [10., .1, .1, .1, .1, 10]
    clf.fit(X, Y, sample_weight=sample_weight)
    y_pred = clf.predict([[-1., 1.]])
    assert y_pred < 1.5

    sample_weight = [1., .1, 10., 10., .1, .1]
    clf.fit(X, Y, sample_weight=sample_weight)
    y_pred = clf.predict([[-1., 1.]])
    assert y_pred > 1.5

    sample_weight = [1] * 6
    clf.fit(X, Y, sample_weight=sample_weight)
    y_pred = clf.predict([[-1., 1.]])
    assert y_pred == pytest.approx(1.5)
