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
import re
import sys
import trace
from collections import namedtuple
from glob import glob

import numpy as np
import pytest
import scipy
import sklearn.utils.validation

from daal4py.sklearn._utils import sklearn_check_version
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


_TRACE_ALLOW_DICT = {
    i: os.path.dirname(importlib.util.find_spec(i).origin)
    for i in ["sklearn", "sklearnex", "onedal", "daal4py"]
}


def _whitelist_to_blacklist():
    """block all standard library, builting or site packages which are not
    related to sklearn, daal4py, onedal or sklearnex"""

    blacklist = []
    for path in sys.path:
        try:
            if any([path in i for i in _TRACE_ALLOW_DICT.values()]):
                blacklist += [
                    f.path
                    for f in os.scandir(path)
                    if f.name not in _TRACE_ALLOW_DICT.keys()
                ]
            else:
                blacklist += [path]
        except FileNotFoundError:
            blacklist += [path]
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
        for modulename, file in _TRACE_ALLOW_DICT.items():
            text = text.replace(file, modulename)
        regex_func = (
            r"(?<=funcname: )\S*(?=\n)"  # needed due to differences in module structure
        )
        regex_mod = r"(?<=--- modulename: )\S*(?=\.py)"  # needed due to differences in module structure

        regex_callingline = r"(.*?)(?:=?\r|\n).*?(?:funcname: ).*"

        cache.set("key", key)
        cache.set(
            "text",
            [
                re.findall(regex_func, text),
                text,
                [i.replace(os.sep, ".") for i in re.findall(regex_mod, text)],
                [""] + re.findall(regex_callingline, text),
            ],
        )

    return cache.get("text", "")


def call_validate_data(text, estimator, method):
    # skip if a daal4py estimator
    if (
        estimator in PATCHED_MODELS
        and PATCHED_MODELS[estimator].__module__.startswith("daal4py")
    ) or (
        estimator in SPECIAL_INSTANCES
        and SPECIAL_INSTANCES[estimator].__module__.startswith("daal4py")
    ):
        pytest.skip("daal4py estimators are not subject to sklearnex design rules")

    if "to_table" not in text[0]:
        pytest.skip("onedal backend not used in this function")

    count = 1 if "fit" not in method else 2
    validate_data = "validate_data" if sklearn_check_version("1.6") else "_validate_data"
    assert (
        text[0].count(validate_data) == count
    ), f"sklearn's f{validate_data} should be called"
    assert (
        text[0].count("_check_n_features") == count
    ), f"estimator should validate n_features_in_"


DESIGN_RULES = []

if sklearn_check_version("1.0"):
    DESIGN_RULES += [call_validate_data]


@pytest.mark.parametrize("design_pattern", DESIGN_RULES)
@pytest.mark.parametrize("estimator, method", gen_models_info(PATCHED_MODELS))
def test_estimator(estimator, method, design_pattern, estimator_trace):
    design_pattern(estimator_trace, estimator, method)
