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

import importlib.util
import os
import pathlib
import pkgutil
import re
import sys
import trace
from glob import glob

import numpy as np
import pytest
import scipy
import sklearn.utils.validation
from sklearn.utils import all_estimators

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

_DESIGN_RULE_VIOLATIONS = {
    "PCA-fit_transform-call_validate_data": "calls both 'fit' and 'transform'",
    "IncrementalEmpiricalCovariance-score-call_validate_data": "must call clone of itself",
    "SVC(probability=True)-fit-call_validate_data": "SVC fit can use sklearn estimator",
    "NuSVC(probability=True)-fit-call_validate_data": "NuSVC fit can use sklearn estimator",
    "LogisticRegression-score-n_jobs_check": "uses daal4py for cpu in sklearnex",
    "LogisticRegression-fit-n_jobs_check": "uses daal4py for cpu in sklearnex",
    "LogisticRegression-predict-n_jobs_check": "uses daal4py for cpu in sklearnex",
    "LogisticRegression-predict_log_proba-n_jobs_check": "uses daal4py for cpu in sklearnex",
    "LogisticRegression-predict_proba-n_jobs_check": "uses daal4py for cpu in sklearnex",
    "KNeighborsClassifier-kneighbors-n_jobs_check": "uses daal4py for cpu in onedal",
    "KNeighborsClassifier-fit-n_jobs_check": "uses daal4py for cpu in onedal",
    "KNeighborsClassifier-score-n_jobs_check": "uses daal4py for cpu in onedal",
    "KNeighborsClassifier-predict-n_jobs_check": "uses daal4py for cpu in onedal",
    "KNeighborsClassifier-predict_proba-n_jobs_check": "uses daal4py for cpu in onedal",
    "KNeighborsClassifier-kneighbors_graph-n_jobs_check": "uses daal4py for cpu in onedal",
    "KNeighborsRegressor-kneighbors-n_jobs_check": "uses daal4py for cpu in onedal",
    "KNeighborsRegressor-fit-n_jobs_check": "uses daal4py for cpu in onedal",
    "KNeighborsRegressor-score-n_jobs_check": "uses daal4py for cpu in onedal",
    "KNeighborsRegressor-predict-n_jobs_check": "uses daal4py for cpu in onedal",
    "KNeighborsRegressor-kneighbors_graph-n_jobs_check": "uses daal4py for cpu in onedal",
    "NearestNeighbors-kneighbors-n_jobs_check": "uses daal4py for cpu in onedal",
    "NearestNeighbors-fit-n_jobs_check": "uses daal4py for cpu in onedal",
    "NearestNeighbors-radius_neighbors-n_jobs_check": "uses daal4py for cpu in onedal",
    "NearestNeighbors-kneighbors_graph-n_jobs_check": "uses daal4py for cpu in onedal",
    "NearestNeighbors-radius_neighbors_graph-n_jobs_check": "uses daal4py for cpu in onedal",
    "LocalOutlierFactor-fit-n_jobs_check": "uses daal4py for cpu in onedal",
    "LocalOutlierFactor-kneighbors-n_jobs_check": "uses daal4py for cpu in onedal",
    "LocalOutlierFactor-kneighbors_graph-n_jobs_check": "uses daal4py for cpu in onedal",
    "KNeighborsClassifier(algorithm='brute')-kneighbors-n_jobs_check": "uses daal4py for cpu in onedal",
    "KNeighborsClassifier(algorithm='brute')-fit-n_jobs_check": "uses daal4py for cpu in onedal",
    "KNeighborsClassifier(algorithm='brute')-score-n_jobs_check": "uses daal4py for cpu in onedal",
    "KNeighborsClassifier(algorithm='brute')-predict-n_jobs_check": "uses daal4py for cpu in onedal",
    "KNeighborsClassifier(algorithm='brute')-predict_proba-n_jobs_check": "uses daal4py for cpu in onedal",
    "KNeighborsClassifier(algorithm='brute')-kneighbors_graph-n_jobs_check": "uses daal4py for cpu in onedal",
    "KNeighborsRegressor(algorithm='brute')-kneighbors-n_jobs_check": "uses daal4py for cpu in onedal",
    "KNeighborsRegressor(algorithm='brute')-fit-n_jobs_check": "uses daal4py for cpu in onedal",
    "KNeighborsRegressor(algorithm='brute')-score-n_jobs_check": "uses daal4py for cpu in onedal",
    "KNeighborsRegressor(algorithm='brute')-predict-n_jobs_check": "uses daal4py for cpu in onedal",
    "KNeighborsRegressor(algorithm='brute')-kneighbors_graph-n_jobs_check": "uses daal4py for cpu in onedal",
    "NearestNeighbors(algorithm='brute')-kneighbors-n_jobs_check": "uses daal4py for cpu in onedal",
    "NearestNeighbors(algorithm='brute')-fit-n_jobs_check": "uses daal4py for cpu in onedal",
    "NearestNeighbors(algorithm='brute')-radius_neighbors-n_jobs_check": "uses daal4py for cpu in onedal",
    "NearestNeighbors(algorithm='brute')-kneighbors_graph-n_jobs_check": "uses daal4py for cpu in onedal",
    "NearestNeighbors(algorithm='brute')-radius_neighbors_graph-n_jobs_check": "uses daal4py for cpu in onedal",
    "LocalOutlierFactor(novelty=True)-fit-n_jobs_check": "uses daal4py for cpu in onedal",
    "LocalOutlierFactor(novelty=True)-kneighbors-n_jobs_check": "uses daal4py for cpu in onedal",
    "LocalOutlierFactor(novelty=True)-kneighbors_graph-n_jobs_check": "uses daal4py for cpu in onedal",
    "LogisticRegression(solver='newton-cg')-score-n_jobs_check": "uses daal4py for cpu in sklearnex",
    "LogisticRegression(solver='newton-cg')-fit-n_jobs_check": "uses daal4py for cpu in sklearnex",
    "LogisticRegression(solver='newton-cg')-predict-n_jobs_check": "uses daal4py for cpu in sklearnex",
    "LogisticRegression(solver='newton-cg')-predict_log_proba-n_jobs_check": "uses daal4py for cpu in sklearnex",
    "LogisticRegression(solver='newton-cg')-predict_proba-n_jobs_check": "uses daal4py for cpu in sklearnex",
}


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


def _sklearnex_walk(func):
    """this replaces checks on pkgutils to look through sklearnex
    folders specifically"""

    def wrap(*args, **kwargs):
        if "prefix" in kwargs and kwargs["prefix"] == "sklearn.":
            kwargs["prefix"] = "sklearnex."
        if "path" in kwargs:
            # force root to sklearnex
            kwargs["path"] = [str(pathlib.Path(__file__).parent.parent)]
        for pkginfo in func(*args, **kwargs):
            # Do not allow spmd to be yielded
            if "spmd" not in pkginfo.name.split("."):
                yield pkginfo

    return wrap


def test_class_trailing_underscore_ban(monkeypatch):
    """Trailing underscores are defined for sklearn to be signatures of a fitted
    estimator instance, sklearnex extends this to the classes as well"""
    monkeypatch.setattr(pkgutil, "walk_packages", _sklearnex_walk(pkgutil.walk_packages))
    estimators = all_estimators()  # list of tuples
    for name, obj in estimators:
        if "preview" not in obj.__module__ and "daal4py" not in obj.__module__:
            # propeties also occur in sklearn, especially in deprecations and are expected
            # to error if queried and the estimator is not fitted
            assert all(
                [
                    isinstance(getattr(obj, attr), property)
                    or (attr.startswith("_") or not attr.endswith("_"))
                    for attr in dir(obj)
                ]
            ), f"{name} contains class attributes which have a trailing underscore but no leading one"


def test_all_estimators_covered(monkeypatch):
    """Check that all estimators defined in sklearnex are available in either the
    patch map or covered in special testing via SPECIAL_INSTANCES. The estimator
    must inherit sklearn's BaseEstimator and must not have a leading underscore.
    The sklearnex.spmd and sklearnex.preview packages are not tested.
    """
    monkeypatch.setattr(pkgutil, "walk_packages", _sklearnex_walk(pkgutil.walk_packages))
    estimators = all_estimators()  # list of tuples
    uncovered_estimators = []
    for name, obj in estimators:
        # do nothing if defined in preview
        if "preview" not in obj.__module__ and not (
            any([issubclass(est, obj) for est in PATCHED_MODELS.values()])
            or any([issubclass(est.__class__, obj) for est in SPECIAL_INSTANCES.values()])
        ):
            uncovered_estimators += [".".join([obj.__module__, name])]

    assert (
        uncovered_estimators == []
    ), f"{uncovered_estimators} are currently not included"


def _fullpath(path):
    return os.path.realpath(os.path.expanduser(path))


_TRACE_ALLOW_DICT = {
    i: _fullpath(os.path.dirname(importlib.util.find_spec(i).origin))
    for i in ["sklearn", "sklearnex", "onedal", "daal4py"]
}


def _whitelist_to_blacklist():
    """block all standard library, built-in or site packages which are not
    related to sklearn, daal4py, onedal or sklearnex"""

    def _commonpath(inp):
        # ValueError generated by os.path.commonpath when it is on a separate drive
        try:
            return os.path.commonpath(inp)
        except ValueError:
            return ""

    blacklist = []
    for path in sys.path:
        fpath = _fullpath(path)
        try:
            # if candidate path is a parent directory to any directory in the whitelist
            if any(
                [_commonpath([i, fpath]) == fpath for i in _TRACE_ALLOW_DICT.values()]
            ):
                # find all sub-paths which are not in the whitelist and block them
                # they should not have a common path that is either the whitelist path
                # or the sub-path (meaning one is a parent directory of the either)
                for f in os.scandir(fpath):
                    temppath = _fullpath(f.path)
                    if all(
                        [
                            _commonpath([i, temppath]) not in [i, temppath]
                            for i in _TRACE_ALLOW_DICT.values()
                        ]
                    ):
                        blacklist += [temppath]
            # add path to blacklist if not a sub path of anything in the whitelist
            elif all([_commonpath([i, fpath]) != i for i in _TRACE_ALLOW_DICT.values()]):
                blacklist += [fpath]
        except FileNotFoundError:
            blacklist += [fpath]
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
        monkeypatch.setattr(trace, "_modname", _fullpath)
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
            {
                "funcs": re.findall(regex_func, text),
                "trace": text,
                "modules": [i.replace(os.sep, ".") for i in re.findall(regex_mod, text)],
                "callingline": [""] + re.findall(regex_callingline, text),
            },
        )

    return cache.get("text", None)


def call_validate_data(text, estimator, method):
    """test that the sklearn function/attribute validate_data is
    called once before offloading to oneDAL in sklearnex"""
    try:
        # get last to_table call showing end of oneDAL input portion of code
        idx = len(text["funcs"]) - 1 - text["funcs"][::-1].index("to_table")
        validfuncs = text["funcs"][:idx]
    except ValueError:
        pytest.skip("onedal backend not used in this function")

    validate_data = "validate_data" if sklearn_check_version("1.6") else "_validate_data"

    assert (
        validfuncs.count(validate_data) == 1
    ), f"sklearn's {validate_data} should be called"
    assert (
        validfuncs.count("_check_feature_names") == 1
    ), "estimator should check feature names in validate_data"


def n_jobs_check(text, estimator, method):
    """verify the n_jobs is being set if '_get_backend' or 'to_table' is called"""
    # remove the _get_backend function from sklearnex from considered _get_backend
    count = max(
        text["funcs"].count("to_table"),
        len(
            [
                i
                for i in range(len(text["funcs"]))
                if text["funcs"][i] == "_get_backend"
                and "sklearnex" not in text["modules"][i]
            ]
        ),
    )
    n_jobs_count = text["funcs"].count("n_jobs_wrapper")

    assert bool(count) == bool(
        n_jobs_count
    ), f"verify if {method} should be in control_n_jobs' decorated_methods for {estimator}"


def runtime_property_check(text, estimator, method):
    """use of Python's 'property' should not be used at runtime, only at class instantiation"""
    assert (
        len(re.findall(r"property\(", text["trace"])) == 0
    ), f"{estimator}.{method} should only use 'property' at instantiation"


DESIGN_RULES = [n_jobs_check, runtime_property_check]


if sklearn_check_version("1.0"):
    DESIGN_RULES += [call_validate_data]


@pytest.mark.parametrize("design_pattern", DESIGN_RULES)
@pytest.mark.parametrize(
    "estimator, method",
    gen_models_info({**PATCHED_MODELS, **SPECIAL_INSTANCES}, fit=True, daal4py=False),
)
def test_estimator(estimator, method, design_pattern, estimator_trace):
    # These tests only apply to sklearnex estimators
    try:
        design_pattern(estimator_trace, estimator, method)
    except AssertionError:
        key = "-".join([estimator, method, design_pattern.__name__])
        if key in _DESIGN_RULE_VIOLATIONS:
            pytest.xfail(_DESIGN_RULE_VIOLATIONS[key])
        else:
            raise
