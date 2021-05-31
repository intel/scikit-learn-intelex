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

from numpy.testing import assert_array_equal
from sklearn import datasets

from onedal.svm import NuSVR


def test_diabetes_simple():
    diabetes = datasets.load_diabetes()
    clf = NuSVR(kernel='linear', C=10.)
    clf.fit(diabetes.data, diabetes.target)
    assert clf.score(diabetes.data, diabetes.target) > 0.02


def test_pickle():
    diabetes = datasets.load_diabetes()
    clf = NuSVR(kernel='rbf', C=10.)
    clf.fit(diabetes.data, diabetes.target)
    expected = clf.predict(diabetes.data)

    import pickle
    dump = pickle.dumps(clf)
    clf2 = pickle.loads(dump)

    assert type(clf2) == clf.__class__
    result = clf2.predict(diabetes.data)
    assert_array_equal(expected, result)
