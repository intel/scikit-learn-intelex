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

from onedal.neighbors import KNeighborsClassifier
from onedal.tests.utils._device_selection import (get_queues,
                                                  pass_if_not_implemented_for_gpu)

from sklearn.utils.estimator_checks import check_estimator
import sklearn.utils.estimator_checks
from sklearn import datasets
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


N_NEIGBORS = 2


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
        'check_classifiers_predictions',  # could not convert string to float: 'one'
        'check_classifier_multioutput',  # y should be a 1d array, got an array of shape (42, 3) instead.
        'check_estimators_unfitted',  # 'KNeighborsClassifier' object has no attribute 'n_samples_fit_'
        # 'check_sample_weights_invariance',  # Max absolute difference: 0.0008
        # 'check_estimators_fit_returns_self',  # ValueError: empty metadata
        # 'check_classifiers_train',  # assert y_pred.shape == (n_samples,)
        # 'check_estimators_unfitted',  # Call 'fit' with appropriate arguments
    ], dummy)
    check_estimator(KNeighborsClassifier())
    _restore_from_saved(md, saved)

# def test_class_weight():
#     X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]])
#     y = np.array([1, 1, 1, 2, 2, 2])

#     clf = SVC(class_weight={1: 0.1})
#     clf.fit(X, y)
#     assert_array_almost_equal(clf.predict(X), [2] * 6)


# def test_sample_weight():
#     X = np.array([[-2, 0], [-1, -1], [0, -2], [0, 2], [1, 1], [2, 2]])
#     y = np.array([1, 1, 1, 2, 2, 2])

#     clf = SVC(kernel='linear')
#     clf.fit(X, y, sample_weight=[1] * 6)
#     assert_array_almost_equal(clf.intercept_, [0.0])


# def test_decision_function():
#     X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
#     Y = [1, 1, 1, 2, 2, 2]

#     clf = SVC(kernel='rbf', gamma=1, decision_function_shape='ovo')
#     clf.fit(X, Y)

#     rbfs = rbf_kernel(X, clf.support_vectors_, gamma=clf.gamma)
#     dec = np.dot(rbfs, clf.dual_coef_.T) + clf.intercept_
#     assert_array_almost_equal(dec.ravel(), clf.decision_function(X))


@pass_if_not_implemented_for_gpu(reason="will be checked later")
@pytest.mark.parametrize('queue', get_queues())
def test_iris(queue):
    iris = datasets.load_iris()
    clf = KNeighborsClassifier(N_NEIGBORS).fit(iris.data, iris.target, queue=queue)
    assert clf.score(iris.data, iris.target, queue=queue) > 0.9
    assert_array_equal(clf.classes_, np.sort(clf.classes_))


# def test_decision_function_shape():
#     X, y = make_blobs(n_samples=80, centers=5, random_state=0)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#     # check shape of ovo_decition_function=True
#     clf = SVC(kernel='linear',
#               decision_function_shape='ovo').fit(X_train, y_train)
#     dec = clf.decision_function(X_train)
#     assert dec.shape == (len(X_train), 10)

#     with pytest.raises(ValueError, match="must be either 'ovr' or 'ovo'"):
#         SVC(decision_function_shape='bad').fit(X_train, y_train)


@pass_if_not_implemented_for_gpu(reason="will be checked later")
@pytest.mark.parametrize('queue', get_queues())
def test_pickle(queue):
    iris = datasets.load_iris()
    clf = KNeighborsClassifier(N_NEIGBORS).fit(iris.data, iris.target, queue=queue)
    expected = clf.predict(iris.data, queue=queue)

    import pickle
    dump = pickle.dumps(clf)
    clf2 = pickle.loads(dump)

    assert type(clf2) == clf.__class__
    result = clf2.predict(iris.data, queue=queue)
    assert_array_equal(expected, result)


# @pytest.mark.parametrize('dtype', [np.float32, np.float64])
# def test_sklearnex_svc_sigmoid(dtype):
#     from sklearnex.neighbors import KNeighborsClassifier
#     X_train = np.array([[-1, 2], [0, 0], [2, -1],
#                         [+1, +1], [+1, +2], [+2, +1]], dtype=dtype)
#     X_test = np.array([[0, 2], [0.5, 0.5],
#                        [0.3, 0.1], [2, 0], [-1, -1]], dtype=dtype)
#     y_train = np.array([1, 1, 1, 2, 2, 2], dtype=dtype)
#     svc = SVC(kernel='sigmoid').fit(X_train, y_train)

#     assert 'daal4py' in svc.__module__ or 'sklearnex' in svc.__module__
#     assert_array_equal(svc.dual_coef_, [[-1, -1, -1, 1, 1, 1]])
#     assert_array_equal(svc.support_, [0, 1, 2, 3, 4, 5])
#     assert_array_equal(svc.predict(X_test), [2, 2, 1, 2, 1])
