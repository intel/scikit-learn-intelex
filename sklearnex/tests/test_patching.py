# ==============================================================================
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
# ==============================================================================

import inspect
import os
import pathlib
import re
import subprocess
import sys
import warnings
from inspect import isclass

import pytest
from _models_info import TO_SKIP
from sklearn.base import BaseEstimator

from sklearnex import get_patch_map, is_patched_instance, patch_sklearn, unpatch_sklearn


def get_branch(s):
    if len(s) == 0:
        return "NO INFO"
    for i in s:
        if "failed to run accelerated version, fallback to original Scikit-learn" in i:
            return "was in OPT, but go in Scikit"
    for i in s:
        if "running accelerated version" in i:
            return "OPT"
    return "Scikit"


def run_parse(mas, result):
    name, dtype = mas[0].split()
    temp = []
    INFO_POS = 16
    for i in range(1, len(mas)):
        mas[i] = mas[i][INFO_POS:]  # remove 'SKLEARNEX INFO: '
        if not mas[i].startswith("sklearn"):
            ind = name + " " + dtype + " " + mas[i]
            result[ind] = get_branch(temp)
            temp.clear()
        else:
            temp.append(mas[i])


def get_result_log():
    os.environ["SKLEARNEX_VERBOSE"] = "INFO"
    absolute_path = str(pathlib.Path(__file__).parent.absolute())
    try:
        process = subprocess.check_output(
            [sys.executable, absolute_path + "/utils/_launch_algorithms.py"]
        )
    except subprocess.CalledProcessError as e:
        print(e)
        exit(1)
    mas = []
    result = {}
    for i in process.decode().split("\n"):
        if i.startswith("SKLEARNEX WARNING"):
            continue
        if not i.startswith("SKLEARNEX INFO") and len(mas) != 0:
            run_parse(mas, result)
            mas.clear()
            mas.append(i.strip())
        else:
            mas.append(i.strip())
    del os.environ["SKLEARNEX_VERBOSE"]
    return result


result_log = get_result_log()


@pytest.mark.parametrize("configuration", result_log)
def test_patching(configuration):
    if "OPT" in result_log[configuration]:
        return
    for skip in TO_SKIP:
        if re.search(skip, configuration) is not None:
            pytest.skip("SKIPPED", allow_module_level=False)
    raise ValueError("Test patching failed: " + configuration)


def _load_all_models(patched):
    if patched:
        patch_sklearn()

    models = {}
    for patch_infos in get_patch_map().values():
        maybe_class = getattr(patch_infos[0][0][0], patch_infos[0][0][1], None)
        if (
            maybe_class is not None
            and isclass(maybe_class)
            and issubclass(maybe_class, BaseEstimator)
        ):
            models[patch_infos[0][0][1]] = maybe_class

    if patched:
        unpatch_sklearn()

    return models


PATCHED_MODELS = _load_all_models(patched=True)
UNPATCHED_MODELS = _load_all_models(patched=False)


@pytest.mark.parametrize("estimator", UNPATCHED_MODELS.keys())
def test_is_patched_instance(estimator):
    patched = PATCHED_MODELS[estimator]
    unpatched = UNPATCHED_MODELS[estimator]
    assert is_patched_instance(patched), f"{patched} is a patched instance"
    assert not is_patched_instance(unpatched), f"{unpatched} is an unpatched instance"


@pytest.mark.parametrize("estimator", PATCHED_MODELS.keys())
def test_docstring_patching_match(estimator):
    patched = PATCHED_MODELS[estimator]
    unpatched = UNPATCHED_MODELS[estimator]
    patched_docstrings = {
        i: getattr(patched, i).__doc__
        for i in dir(patched)
        if not i.startswith("_") and not i.endswith("_")
    }
    unpatched_docstrings = {
        i: getattr(unpatched, i).__doc__
        for i in dir(unpatched)
        if not i.startswith("_") and not i.endswith("_")
    }

    # check class docstring match if a docstring is available
    assert patched.__doc__ is not None or unpatched.__doc__ is None
    if patched.__doc__ != unpatched.__doc__:
        warnings.warn(
            f"class {estimator} has a custom docstring which does not match sklearn"
        )

    # check class attribute docstrings
    for i in unpatched_docstrings:
        assert patched_docstrings[i] is not None or unpatched_docstrings[i] is None
        if patched_docstrings[i] != unpatched_docstrings[i]:
            warnings.warn(
                f"{estimator}.{i} has a custom docstring which does not match sklearn"
            )


@pytest.mark.parametrize("member", ["_onedal_cpu_supported", "_onedal_gpu_supported"])
@pytest.mark.parametrize(
    "name",
    [i for i in PATCHED_MODELS.keys() if "sklearnex" in PATCHED_MODELS[i].__module__],
)
def test_onedal_supported_member(name, member):
    patched = PATCHED_MODELS[name]
    sig = str(inspect.signature(getattr(patched, member)))
    assert "(self, method_name, *data)" == sig
