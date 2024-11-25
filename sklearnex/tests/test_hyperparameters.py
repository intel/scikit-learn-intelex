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


import pytest

from sklearnex._utils import register_hyperparameters


def test_register_hyperparameters():
    hyperparameters_map = {"op": "hyperparameters"}

    @register_hyperparameters(hyperparameters_map)
    class Test:
        pass

    # assert the correct value is returned
    assert Test.get_hyperparameters("op") == "hyperparameters"


def test_register_hyperparameters_issues_warning():
    hyperparameters_map = {"op": "hyperparameters"}

    @register_hyperparameters(hyperparameters_map)
    class Test:
        pass

    # assert a warning is issued when trying to modify the hyperparameters per instance
    with pytest.warns(Warning):
        Test().get_hyperparameters("op")
