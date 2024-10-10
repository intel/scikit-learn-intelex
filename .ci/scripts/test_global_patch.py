#! /usr/bin/env python
# ===============================================================================
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
# ===============================================================================

import subprocess
import sys

import pytest


@pytest.fixture
def patch_svc_from_command_line(request):
    err_code = subprocess.call(
        [sys.executable, "-m", "sklearnex.glob", "patch_sklearn", "-a", "svc"]
    )
    assert not err_code

    def unpatch_from_cmd():
        err_code = subprocess.call(
            [sys.executable, "-m", "sklearnex.glob", "unpatch_sklearn"]
        )
        assert not err_code

    request.addfinalizer(unpatch_from_cmd)
    return


def test_patching_svc_from_command_line(patch_svc_from_command_line):
    from sklearn.svm import SVC, SVR

    assert SVC.__module__.startswith("daal4py") or SVC.__module__.startswith("sklearnex")
    assert not SVR.__module__.startswith("daal4py") and not SVR.__module__.startswith(
        "sklearnex"
    )


def test_unpatching_svc_from_command_line(patch_svc_from_command_line):
    err_code = subprocess.call(
        [sys.executable, "-m", "sklearnex.glob", "unpatch_sklearn"]
    )
    assert not err_code
    from sklearnex import unpatch_sklearn

    unpatch_sklearn()
    from sklearn.svm import SVC, SVR

    assert not SVC.__module__.startswith("daal4py") and not SVC.__module__.startswith(
        "sklearnex"
    )
    assert not SVR.__module__.startswith("daal4py") and not SVR.__module__.startswith(
        "sklearnex"
    )


@pytest.fixture
def patch_svc_from_function(request):
    from sklearnex import patch_sklearn, unpatch_sklearn

    patch_sklearn(name=["svc"], global_patch=True)

    def unpatch_from_fn():
        unpatch_sklearn(global_unpatch=True)

    request.addfinalizer(unpatch_from_fn)
    return


def test_patching_svc_from_function(patch_svc_from_function):
    from sklearn.svm import SVC, SVR

    assert SVC.__module__.startswith("daal4py") or SVC.__module__.startswith("sklearnex")
    assert not SVR.__module__.startswith("daal4py") and not SVR.__module__.startswith(
        "sklearnex"
    )


def test_unpatching_svc_from_function(patch_svc_from_function):
    from sklearnex import unpatch_sklearn

    unpatch_sklearn(global_unpatch=True)
    from sklearn.svm import SVC, SVR

    assert not SVC.__module__.startswith("daal4py") and not SVC.__module__.startswith(
        "sklearnex"
    )
    assert not SVR.__module__.startswith("daal4py") and not SVR.__module__.startswith(
        "sklearnex"
    )
