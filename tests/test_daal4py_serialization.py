# ==============================================================================
# Copyright 2024 Intel Corporation
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
# ==============================================================================

import pickle
import unittest

import numpy as np

import daal4py


class Test(unittest.TestCase):
    def test_serialization_of_algorithms(self):
        obj_original = daal4py.qr(fptype="float")
        obj_deserialized = pickle.loads(pickle.dumps(obj_original))

        rng = np.random.default_rng(seed=123)
        X = rng.standard_normal(size=(10, 5))

        Q_orig = obj_original.compute(X).matrixQ
        Q_deserialized = obj_deserialized.compute(X).matrixQ
        np.testing.assert_almost_equal(Q_orig, Q_deserialized)
        assert Q_orig.dtype == Q_deserialized.dtype
