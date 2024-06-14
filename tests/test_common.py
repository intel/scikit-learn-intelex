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

import importlib
import os
from glob import glob

TARGET_OFFLOAD_ALLOWED_LOCATIONS = [
    "_config.py",
    "_device_offload.py",
    "test",
    "svc.py",
    "svm" + os.sep + "_common.py",
]


def _test_primitive_usage_ban(primimtive_name, banned_locations, allowed_locations=None):
    """This test blocks the usage of the primitive in
    in certain files.
    """

    loc = importlib.import_module(banned_locations).__file__

    path = loc.replace("__init__.py", "")
    files = [y for x in os.walk(path) for y in glob(os.path.join(x[0], "*.py"))]

    output = []

    for f in files:
        if open(f, "r").read().find(primimtive_name) != -1:
            output += [f.replace(path, banned_locations + os.sep)]

    # remove this file from the list
    if allowed_locations:
        for allowed in allowed_locations:
            output = [i for i in output if allowed not in i]

    return output


def test_target_offload_ban():
    """This test blocks the use of target_offload in
    in sklearnex files. Offloading computation to devices
    via target_offload should only occur externally, and not
    within the architecture of the sklearnex classes. This
    is for clarity, traceability and maintainability.
    """
    output = _test_primitive_usage_ban(
        primimtive_name="target_offload",
        banned_locations="sklearnex",
        allowed_locations=TARGET_OFFLOAD_ALLOWED_LOCATIONS,
    )
    output = "\n".join(output)
    assert output == "", f"target offloading is occuring in: \n{output}"


def test_sklearn_check_version_ban():
    """This test blocks the use of sklearn_check_version
    in onedal files. The versioning should occur in the
    sklearnex package for clarity and maintainability.
    """
    output = _test_primitive_usage_ban(
        primimtive_name="sklearn_check_version", banned_locations="onedal"
    )

    # remove this file from the list
    output = "\n".join([i for i in output if "test_common.py" not in i])
    assert output == "", f"sklearn versioning is occuring in: \n{output}"


def test_sklearn_check_version_ban_1():
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
