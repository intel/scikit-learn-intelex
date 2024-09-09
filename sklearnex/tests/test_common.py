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

import logging
import os
from glob import glob

import pytest
import sklearn.utils.validation

import daal4py.utils.validation
from sklearnex.tests._utils import (
    PATCHED_MODELS,
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


def debug_function(name, func, logger, *args, **kwargs):
    """This wraps a function to make it verbose for analysis,
    it will print the name of the function and its location to
    the specified logger at debug level. This should use an
    alternate logger (not sklearnex) to avoid interaction with
    other logging functionality"""
    if logger == "sklearnex":
        raise ValueError("sklearnex logger is protected")
    log = logging.getLogger(logger)

    def wrapped_func(*args, **kwargs):
        log.debug(name + " " + ".".join(func.__module__, __name__))
        return func(*args, **kwargs)

    return wrapped_func


class LogEstimator(object):
    """ wrap sklearnex estimator to test for operational
    design conformance by logging"""

    def test_validate_params():

    def test_finite_checking():
