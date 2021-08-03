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


def get_patch_message(s):
    import sys
    if s == "onedal":
        message = "running accelerated version on "
        if 'daal4py.oneapi' in sys.modules:
            from daal4py.oneapi import _get_device_name_sycl_ctxt
            dev = _get_device_name_sycl_ctxt()
            if dev == 'cpu' or dev == 'host':
                message += 'CPU'
            elif dev == 'gpu':
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


def get_patch_str(name=None, verbose=True):
    return f"""try:
    from sklearnex import patch_sklearn
    patch_sklearn(name={str(name)}, verbose={str(verbose)})
    del patch_sklearn
except (ImportError, ModuleNotFoundError):
    pass"""


def get_patch_str_re():
    return r"""\ntry:
    from sklearnex import patch_sklearn
    patch_sklearn\(name=.*, verbose=.*\)
    del patch_sklearn
except \(ImportError, ModuleNotFoundError\):
    pass\n"""


def patch_sklearn_global(name=None, verbose=True):
    import os
    import re
    try:
        import sklearn
    except ImportError:
        raise ImportError("Scikit-learn could not be imported. Nothing to patch\n")

    init_file_path = sklearn.__file__
    distributor_file_path = os.path.join(os.path.dirname(init_file_path),
                                         "_distributor_init.py")

    with open(distributor_file_path, 'r', encoding='utf-8') as distributor_file:
        lines = distributor_file.read()
        if re.search(get_patch_str_re(), lines):
            lines = re.sub(get_patch_str_re(), '', lines)

    with open(distributor_file_path, 'w', encoding='utf-8') as distributor_file:
        distributor_file.write(lines + "\n" + get_patch_str(name, verbose) + "\n")
        print("Scikit-learn successfully patched by Intel(R) Extension for Scikit-learn")
        return


def unpatch_sklearn_global():
    import os
    import re
    try:
        import sklearn
    except ImportError:
        raise ImportError("Scikit-learn could not be imported. Nothing to unpatch\n")

    init_file_path = sklearn.__file__
    distributor_file_path = os.path.join(os.path.dirname(init_file_path),
                                         "_distributor_init.py")

    with open(distributor_file_path, 'r', encoding='utf-8') as distributor_file:
        lines = distributor_file.read()
        if not re.search(get_patch_str_re(), lines):
            print("Scikit-learn didn't patched. Nothing to unpatch\n")
            return
        lines = re.sub(get_patch_str_re(), '', lines)

    with open(distributor_file_path, 'w', encoding='utf-8') as distributor_file:
        distributor_file.write(lines)
        print("Scikit-learn successfully unpatched")
        return
