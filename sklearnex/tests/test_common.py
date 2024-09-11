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
import sys
import trace
from glob import glob

import numpy as np
import scipy
import pytest
import sklearn.utils.validation

from sklearnex.tests._utils import (
    PATCHED_MODELS,
    SPECIAL_INSTANCES,
    call_method,
    gen_dataset,
    gen_models_info,
)

ALLOWED_LOCATIONS = [
    "_config.py",
    "_device_offload.py",
    "test",
    "svc.py",
    "svm" + os.sep + "_common.py",
]


def test_target_offload_ban():
    """This test blocks the use of target_offload in
    in sklearnex files. Offloading computation to devices
    via target_offload should only occur externally, and not
    within the architecture of the sklearnex classes. This
    is for clarity, traceability and maintainability.
    """
    from sklearnex import __file__ as loc

    path = loc.replace("__init__.py", "")
    files = [y for x in os.walk(path) for y in glob(os.path.join(x[0], "*.py"))]

    output = []

    for f in files:
        if open(f, "r").read().find("target_offload") != -1:
            output += [f.replace(path, "sklearnex" + os.sep)]

    # remove this file from the list
    for allowed in ALLOWED_LOCATIONS:
        output = [i for i in output if allowed not in i]

    output = "\n".join(output)
    assert output == "", f"sklearn versioning is occuring in: \n{output}"


def _whitelist_to_blacklist():
    """block all standard library, builting or site packages which are not
    related to sklearn, daal4py, onedal or sklearnex"""

    whitelist = ["sklearn", "sklearnex", "onedal", "daal4py"]
    blacklist = []
    for path in sys.path:
        try:
            if "site-packages" in path or "dist-packages" in path:
                blacklist += [f.path for f in os.scandir(path) if f.name not in whitelist]
            else:
                blacklist += [path]
        except FileNotFoundError:
            pass
    return blacklist


_TRACE_BLOCK_LIST = _whitelist_to_blacklist()


@pytest.fixture
def estimator_trace(estimator, method, cache, capsys, monkeypatch):
    """generate data only once, and only if the key doesn't match"""
    key = "-".join((str(estimator), method))
    flag = cache.get("key", "") != key
    if flag:
        # get estimator
        try:
            est = PATCHED_MODELS[estimator]()
        except IndexError:
            est = SPECIAL_INSTANCES[estimator]

        # get dataset
        X, y = gen_dataset(est)[0]
        # fit dataset if method does not contain 'fit'
        if "fit" not in method:
            est.fit(X, y)

        # initialize tracer to have a more verbose module naming
        # this impacts ignoremods, but it is not used.
        monkeypatch.setattr(trace, "_modname", lambda x: x)
        tracer = trace.Trace(
            count=0,
            trace=1,
            ignoredirs=_TRACE_BLOCK_LIST,
        )
        # call trace on method with dataset
        tracer.runfunc(call_method, est, method, X, y)

        # collect trace for analysis
        text = capsys.readouterr().out
        cache.set("key", key)
        cache.set("text", text)

    return cache.get("text", "")


def assert_all_finite_onedal(text, estimator, method):
    # find the number of inputs into the object

    # find call_method

    # acquire what data is set to

    #

    # if fit in method, give it additional checks

    # if fit check for __init__ and onedal
    # collected all _assert_all_finite calls

    # stop when to_table observed
    text = "funcname: _assert_all_finite"

    # regex _assert_all_finite and find line numbers

    # verify that the onedal

    print(estimator, method)
    print(text)
    assert False


DESIGN_RULES = [
    assert_all_finite_onedal,
]


@pytest.mark.parametrize("design_pattern", DESIGN_RULES)
@pytest.mark.parametrize("estimator, method", gen_models_info(PATCHED_MODELS))
def test_estimator(estimator, method, design_pattern, estimator_trace):
    design_pattern(estimator_trace, estimator, method)
