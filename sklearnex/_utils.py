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

import logging
import os
import warnings
from abc import ABC

from daal4py.sklearn._utils import (
    PatchingConditionsChain as daal4py_PatchingConditionsChain,
)
from daal4py.sklearn._utils import daal_check_version


class StaticOnlyMethod:
    """Descriptor for static methods only. Raises an exception if called on an instance.

    Parameters
    ----------
    func : callable
        The function to be decorated.
    instance_call_behavior : None or Exception or Warning, default=None
        The behavior when the method is called on an instance.
        If None, AttributeError is raised.
        If an Exception or Warning, it is raised or warned respectively.
    """

    def __init__(self, func=None, instance_call_behavior=None):
        self.func = func
        if instance_call_behavior is None:
            self.on_instance_call = AttributeError(
                "This method can only be called on the class, not on an instance."
            )
        elif isinstance(instance_call_behavior, (Exception, Warning)):
            self.on_instance_call = instance_call_behavior
        else:
            raise ValueError(
                f"Invalid input - expected None or an Exception, got {instance_call_behavior}"
            )

    def __call__(self, func):
        self.func = func
        return self

    def __get__(self, instance, _):
        if instance is not None:
            if isinstance(self.on_instance_call, Warning):
                warnings.warn(self.on_instance_call)
            else:
                raise self.on_instance_call

        return self.func


class PatchingConditionsChain(daal4py_PatchingConditionsChain):
    def get_status(self):
        return self.patching_is_enabled

    def write_log(self, queue=None, transferred_to_host=True):
        if self.patching_is_enabled:
            self.logger.info(
                f"{self.scope_name}: {get_patch_message('onedal', queue=queue, transferred_to_host=transferred_to_host)}"
            )
        else:
            self.logger.debug(
                f"{self.scope_name}: debugging for the patch is enabled to track"
                " the usage of Intel® oneAPI Data Analytics Library (oneDAL)"
            )
            for message in self.messages:
                self.logger.debug(
                    f"{self.scope_name}: patching failed with cause - {message}"
                )
            self.logger.info(
                f"{self.scope_name}: {get_patch_message('sklearn', transferred_to_host=transferred_to_host)}"
            )


def set_sklearn_ex_verbose():
    log_level = os.environ.get("SKLEARNEX_VERBOSE")

    logger = logging.getLogger("sklearnex")
    logging_channel = logging.StreamHandler()
    logging_formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
    logging_channel.setFormatter(logging_formatter)
    logger.addHandler(logging_channel)

    try:
        if log_level is not None:
            logger.setLevel(log_level)
    except Exception:
        warnings.warn(
            'Unknown level "{}" for logging.\n'
            'Please, use one of "CRITICAL", "ERROR", '
            '"WARNING", "INFO", "DEBUG".'.format(log_level)
        )


def get_patch_message(s, queue=None, transferred_to_host=True):
    if s == "onedal":
        message = "running accelerated version on "
        if queue is not None:
            if queue.sycl_device.is_gpu:
                message += "GPU"
            elif queue.sycl_device.is_cpu:
                message += "CPU"
            else:
                raise RuntimeError("Unsupported device")
        else:
            message += "CPU"
    elif s == "sklearn":
        message = "fallback to original Scikit-learn"
    elif s == "sklearn_after_onedal":
        message = "failed to run accelerated version, fallback to original Scikit-learn"
    else:
        raise ValueError(
            f"Invalid input - expected one of 'onedal','sklearn',"
            f" 'sklearn_after_onedal', got {s}"
        )
    if transferred_to_host:
        message += (
            ". All input data transferred to host for further backend computations."
        )
    return message


def get_sklearnex_version(rule):
    return daal_check_version(rule)


def register_hyperparameters(hyperparameters_map):
    """Decorator for hyperparameters support in estimator class.
    Adds `get_hyperparameters` method to class.
    """

    def decorator(cls):
        """Add `get_hyperparameters()` static method"""

        @StaticOnlyMethod(
            instance_call_behavior=Warning(
                "Hyperparameters are static variables and can not be modified per instance."
            )
        )
        def get_hyperparameters(op):
            return hyperparameters_map[op]

        cls.get_hyperparameters = get_hyperparameters
        return cls

    return decorator


# This abstract class is meant to generate a clickable doc link for classses
# in sklearnex that are not part of base scikit-learn. It should be inherited
# before inheriting from a scikit-learn estimator, otherwise will get overriden
# by the estimator's original.
class IntelEstimator(ABC):
    @property
    def _doc_link_module(self) -> str:
        return "sklearnex"

    @property
    def _doc_link_template(self) -> str:
        module_path, _ = self.__class__.__module__.rsplit(".", 1)
        class_name = self.__class__.__name__
        return f"https://intel.github.io/scikit-learn-intelex/latest/non-scikit-algorithms.html#{module_path}.{class_name}"
