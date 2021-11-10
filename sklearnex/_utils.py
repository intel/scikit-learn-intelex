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

def set_sklearn_ex_verbose():
    import logging
    import warnings
    import os
    import sys
    logLevel = os.environ.get("SKLEARNEX_VERBOSE")
    try:
        if logLevel is not None:
            logging.basicConfig(
                stream=sys.stdout,
                format='SKLEARNEX %(levelname)s: %(message)s', level=logLevel.upper())
    except Exception:
        warnings.warn('Unknown level "{}" for logging.\n'
                      'Please, use one of "CRITICAL", "ERROR", '
                      '"WARNING", "INFO", "DEBUG".'.format(logLevel))


def get_patch_message(s, queue=None, cpu_fallback=False):
    import sys
    if s == "onedal":
        message = "running accelerated version on "
        if queue is not None:
            if queue.sycl_device.is_gpu:
                message += 'GPU'
            elif queue.sycl_device.is_cpu or queue.sycl_device.is_host:
                message += 'CPU'
            else:
                raise RuntimeError('Unsupported device')

        elif 'daal4py.oneapi' in sys.modules:
            from daal4py.oneapi import _get_device_name_sycl_ctxt
            dev = _get_device_name_sycl_ctxt()
            if dev == 'cpu' or dev == 'host' or dev is None:
                message += 'CPU'
            elif dev == 'gpu':
                if cpu_fallback:
                    message += 'CPU'
                else:
                    message += 'GPU'
            else:
                raise ValueError(f"Unexpected device name {dev}."
                                 " Supported types are host, cpu and gpu")
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
    from daal4py.sklearn._utils import daal_check_version
    return daal_check_version(rule)
