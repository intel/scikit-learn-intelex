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
import daal4py.sklearn


class MonkeyPTest(unittest.TestCase):
    def test_monkey_patching(self):
        _tokens = daal4py.sklearn.sklearn_patch_names()
        self.assertTrue(isinstance(_tokens, list) and len(_tokens) > 0)
        for t in _tokens:
            daal4py.sklearn.unpatch_sklearn(t)
        for t in _tokens:
            daal4py.sklearn.patch_sklearn(t)

        import sklearn
        for a in [(sklearn.decomposition, 'PCA'),
                  (sklearn.linear_model, 'Ridge'),
                  (sklearn.linear_model, 'LinearRegression'),
                  (sklearn.cluster, 'KMeans'),
                  (sklearn.svm, 'SVC'),]:
            class_module = getattr(a[0], a[1]).__module__
            self.assertTrue(class_module.startswith('daal4py'))
