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

import numpy as np
from numpy.testing import assert_array_equal

from onedal.svm import NuSVC

from sklearn import datasets
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


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
