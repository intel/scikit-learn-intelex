#!/usr/bin/env python
#===============================================================================
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
#===============================================================================

import logging
import warnings
import os
import sys
from daal4py.sklearn._utils import daal_check_version


def set_sklearn_ex_verbose():
    log_level = os.environ.get("SKLEARNEX_VERBOSE")

    logger = logging.getLogger('sklearnex')
    logging_channel = logging.StreamHandler()
    logging_formatter = logging.Formatter('%(levelname)s:%(name)s: %(message)s')
    logging_channel.setFormatter(logging_formatter)
    logger.addHandler(logging_channel)

    try:
        if log_level is not None:
            logger.setLevel(log_level)
    except Exception:
        warnings.warn('Unknown level "{}" for logging.\n'
                      'Please, use one of "CRITICAL", "ERROR", '
                      '"WARNING", "INFO", "DEBUG".'.format(log_level))


def get_patch_message(s, queue=None, cpu_fallback=False):
    if s == "onedal":
        message = "running accelerated version on "
        if queue is not None:
            if queue.sycl_device.is_gpu:
                message += 'GPU'
            elif queue.sycl_device.is_cpu:
                message += 'CPU'
            else:
                raise RuntimeError('Unsupported device')

        elif 'daal4py.oneapi' in sys.modules:
            from daal4py.oneapi import _get_device_name_sycl_ctxt
            dev = _get_device_name_sycl_ctxt()
            if dev == 'cpu' or dev is None:
                message += 'CPU'
            elif dev == 'gpu':
                if cpu_fallback:
                    message += 'CPU'
                else:
                    message += 'GPU'
            else:
                raise ValueError(f"Unexpected device name {dev}."
                                 " Supported types are cpu and gpu")
        else:
            message += 'CPU'

    elif s == "sklearn":
        message = "fallback to original Scikit-learn"
    elif s == "sklearn_after_onedal":
        message = "failed to run accelerated version, fallback to original Scikit-learn"
    else:
        raise ValueError(
            f"Invalid input - expected one of 'onedal','sklearn',"
            f" 'sklearn_after_onedal', got {s}")
    return message


def get_sklearnex_version(rule):
    return daal_check_version(rule)
