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


@pytest.fixture
def estimator_trace(estimator, method, cache, capsys):
    key = "-".join((str(estimator), method))
    text = cache.get(key, None)
    if text is None:

        # get estimator
        try:
            est = PATCHED_MODELS[estimator]()
        except IndexError:
            est = SPECIAL_INSTANCES[estimator]

        # get dataset
        [X, y] = gen_dataset(est)
        # fit dataset if method does not contain 'fit'
        if "fit" not in method:
            est.fit(X, y)

        # initialize tracer
        tracer = trace.Trace(count=0, trace=1, ignoremods=(np, scipy, pytest))
        # call trace on method with dataset
        tracer.runfunc(call_method, est, method, X, y)

        # collect trace for analysis
        text = capsys.readouterr().out
        cache.set(key, text)

    return text


def assert_finite_in_onedal(text, estimator, method):
    print("hello")
    print(estimator, method)
    print(text)
    assert False


DESIGN_RULES = [
    assert_finite_in_onedal,
]


@pytest.mark.parametrize("estimator, method", gen_models_info(PATCHED_MODELS))
@pytest.mark.parametrize("design_pattern", DESIGN_RULES, indirect=True)
def test_estimator(estimator, method, design_pattern, estimator_trace):
    design_pattern(estimator_trace, estimator, method)
