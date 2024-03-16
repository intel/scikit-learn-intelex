# ==============================================================================
# Copyright 2023 Intel Corporation
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

from daal4py.sklearn import _utils


@pytest.mark.parametrize(
    "required_version,expected",
    [
        # wrong status -> False
        ((2019, "A", 1), False),
        ((2019, "Z", 100), False),
        ((2020, "A", 100), False),
        ((2020, "Z", 100), False),
        ((2021, "Z", 200), False),
        # major < 2020 -> True
        ((2019, "P", 100), True),
        ((2019, "P", 200), True),
        ((2019, "P", 200), True),
        (((2019, "P", 100), (2018, "P", 100)), True),
        (((2019, "P", 100), (2018, "P", 100)), True),
        # major > 2020 -> False
        ((2021, "P", 100), False),
        (((2021, "P", 100), (2022, "P", 100)), False),
        # major == 2020, patch > 100 -> False
        ((2020, "P", 200), False),
        # major == 2020, patch < 100 -> False
        ((2020, "P", 1), True),
        # major == 2020, patch == 100 -> True (exact version match)
        ((2020, "P", 100), True),
        (((2020, "P", 100), (2021, "Z", 200)), True),
    ],
)
def test_daal_check_version(required_version, expected):
    actual = _utils.daal_check_version(required_version, (2020, "P", 100))
    assert actual == expected, f"{required_version=}, {expected=}, {actual=}"
