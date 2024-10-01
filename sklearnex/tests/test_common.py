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
from sklearnex.tests.utils import (
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

_DESIGN_RULE_VIOLATIONS = [
    "PCA-fit_transform-call_validate_data",  #  calls both "fit" and "transform"
    "IncrementalEmpiricalCovariance-score-call_validate_data",  #  must call clone of itself
    "SVC(probability=True)-fit-call_validate_data",  #  SVC fit can use sklearn estimator
    "NuSVC(probability=True)-fit-call_validate_data",  #  NuSVC fit can use sklearn estimator
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
    """Generate a trace of all function calls in calling estimator.method with cache.

    Parameters
    ----------
    estimator : str
        name of estimator which is a key from PATCHED_MODELS or

    method : str
        name of estimator method which is to be traced and stored

    cache: pytest.fixture (standard)

    capsys: pytest.fixture (standard)

    monkeypatch: pytest.fixture (standard)

    Returns
    -------
    dict: [calledfuncs, tracetext, modules, callinglines]
        Returns a list of important attributes of the trace.
        calledfuncs is the list of called functions, tracetext is the
        total text output of the trace as a string, modules are the
        module locations  of the called functions (must be from daal4py,
        onedal, sklearn, or sklearnex), and callinglines is the line
        which calls the function in calledfuncs
    """
    key = "-".join((str(estimator), method))
    flag = cache.get("key", "") != key
    if flag:
        # get estimator
        try:
            est = PATCHED_MODELS[estimator]()
        except KeyError:
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

        regex_callingline = r"(?<=\n)\S.*(?=\n --- modulename: )"

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

    return cache.get("text", None)


def call_validate_data(text, estimator, method):
    """test that the sklearn function/attribute validate_data is
    called once before offloading to oneDAL in sklearnex"""
    try:
        # get last to_table call showing end of oneDAL input portion of code
        idx = len(text[0]) - 1 - text[0][::-1].index("to_table")
        validfuncs = text[0][:idx]
    except ValueError:
        pytest.skip("onedal backend not used in this function")

    validate_data = "validate_data" if sklearn_check_version("1.6") else "_validate_data"
    try:
        assert (
            validfuncs.count(validate_data) == 1
        ), f"sklearn's {validate_data} should be called"
        assert (
            validfuncs.count("_check_feature_names") == 1
        ), "estimator should check feature names in validate_data"
    except AssertionError:
        if "-".join([estimator, method, "call_validate_data"]) in _DESIGN_RULE_VIOLATIONS:
            pytest.xfail("Allowed violation of design rules")
        else:
            raise


def n_jobs_check(text, estimator, method):
    """verify the n_jobs is being set if '_get_backend' or 'to_table' is called"""
    count = max([text[0].count(name) for name in ["to_table", "_get_backend"]])
    n_jobs_count = text[0].count("n_jobs_wrapper")

    assert bool(count) == bool(
        n_jobs_count
    ), f"verify if {method} should be in control_n_jobs' decorated_methods for {estimator}"


DESIGN_RULES = [n_jobs_check]


if sklearn_check_version("1.0"):
    DESIGN_RULES += [call_validate_data]


@pytest.mark.parametrize("design_pattern", DESIGN_RULES)
@pytest.mark.parametrize(
    "estimator, method",
    gen_models_info({**PATCHED_MODELS, **SPECIAL_INSTANCES}, fit=True, daal4py=False),
)
def test_estimator(estimator, method, design_pattern, estimator_trace):
    # These tests only apply to sklearnex estimators
    design_pattern(estimator_trace, estimator, method)
