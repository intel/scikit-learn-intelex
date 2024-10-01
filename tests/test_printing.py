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

import re
import unittest

import numpy as np

import daal4py


class Test(unittest.TestCase):
    def test_qr_printing(self):
        rng = np.random.default_rng(seed=123)
        X = rng.standard_normal(size=(10, 5))
        qr_algorithm = daal4py.qr()
        qr_result = qr_algorithm.compute(X)
        qr_result_str, qr_result_repr = qr_result.__str__(), qr_result.__repr__()
        assert "matrixQ" in qr_result_str
        assert "matrixR" in qr_result_str
        assert "matrixQ" in qr_result_repr
        assert "matrixR" in qr_result_repr
