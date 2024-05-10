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

import os
from glob import glob

import pytest


def test_sklearn_check_version_ban():
    """This test blocks the use of sklearn_check_version
    in onedal files. The versioning should occur in the
    sklearnex package for clarity and maintainability.
    """
    from onedal import __file__ as loc

    path = loc.replace("__init__.py", "")
    files = [y for x in os.walk(path) for y in glob(os.path.join(x[0], "*.py"))]

    output = []

    for f in files:
        if open(f, "r").read().find("sklearn_check_version") != -1:
            output += [f.replace(path, "onedal" + os.sep)]

    # remove this file from the list
    output = "\n".join([i for i in output if "test_common.py" not in i])
    assert output == "", f"sklearn versioning is occuring in: \n{output}"
