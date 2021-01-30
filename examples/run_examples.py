#===============================================================================
# Copyright 2014-2021 Intel Corporation
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

import os
import struct
import subprocess
import sys

from daal4py import __has_dist__
from daal4py.sklearn._utils import get_daal_version

print('Starting examples validation')
# First item is major version - 2021,
# second is minor+patch - 0110,
# third item is status - B
print('DAAL version:', get_daal_version())

from os.path import join as jp
from time import gmtime, strftime
from collections import defaultdict

exdir = os.path.dirname(os.path.realpath(__file__))

IS_WIN = False
IS_MAC = False
IS_LIN = False
system_os = "not_supported"
if 'linux' in sys.platform:
    IS_LIN = True
    system_os = "lnx"
elif sys.platform == 'darwin':
    IS_MAC = True
    system_os = "mac"
elif sys.platform in ['win32', 'cygwin']:
    IS_WIN = True
    system_os = "win"
else:
    assert False, sys.platform + ' not supported'

assert 8 * struct.calcsize('P') in [32, 64]

if 8 * struct.calcsize('P') == 32:
    logdir = jp(exdir, '_results', 'ia32')
else:
    logdir = jp(exdir, '_results', 'intel64')

availabe_devices = []

try:
    from daal4py.oneapi import sycl_context
    sycl_extention_available = True
except:
    sycl_extention_available = False

if sycl_extention_available:
    try:
        with sycl_context('gpu'):
            gpu_available = True
            availabe_devices.append("gpu")
    except:
        gpu_available = False
    availabe_devices.append("host")
    availabe_devices.append("cpu")


def check_version(rule, target):
    if not isinstance(rule[0], type(target)):
        if rule > target:
            return False
    else:
        for rule_item in rule:
            if rule_item > target:
                return False
            if rule_item[0] == target[0]:
                break
    return True


def check_device(rule, target):
    for rule_item in rule:
        if rule_item not in target:
            return False
    return True


def check_os(rule, target):
    for rule_item in rule:
        if rule_item not in target:
            return False
    return True


def check_library(rule):
    for rule_item in rule:
        try:
            import importlib
            importlib.import_module(rule_item, package=None)
        except ImportError:
            return False
    return True


req_version = defaultdict(lambda: (2019, 'P', 0))
req_version['sycl/dbscan_batch.py'] = \
    (2021, 'P', 100)  # hangs in beta08, need to be fixed
req_version['sycl/linear_regression_batch.py'] = \
    (2021, 'P', 100)  # hangs in beta08, need to be fixed
req_version['sycl/kmeans_batch.py'] = \
    (2021, 'P', 200)  # not equal results for host and gpu runs
req_version['sycl/pca_transform_batch.py'] = (2021, 'P', 200)

req_device = defaultdict(lambda: [])
req_device['sycl/gradient_boosted_regression_batch.py'] = ["gpu"]

req_library = defaultdict(lambda: [])
req_library['gbt_cls_model_create_from_lightgbm_batch.py'] = ['lightgbm']
req_library['gbt_cls_model_create_from_xgboost_batch.py'] = ['xgboost']

req_os = defaultdict(lambda: [])


def get_exe_cmd(ex, nodist, nostream):
    if os.path.dirname(ex).endswith("sycl"):
        if not sycl_extention_available:
            return None
        if not check_version(req_version["sycl/" + os.path.basename(ex)],
                             get_daal_version()):
            return None
        if not check_device(req_device["sycl/" + os.path.basename(ex)], availabe_devices):
            return None
        if not check_os(req_os["sycl/" + os.path.basename(ex)], system_os):
            return None

    if os.path.dirname(ex).endswith("examples"):
        if not check_version(req_version[os.path.basename(ex)], get_daal_version()):
            return None
        if not check_library(req_library[os.path.basename(ex)]):
            return None
    if any(ex.endswith(x) for x in ['batch.py', 'stream.py']):
        return '"' + sys.executable + '" "' + ex + '"'
    if not nostream and ex.endswith('streaming.py'):
        return '"' + sys.executable + '" "' + ex + '"'
    if not nodist and ex.endswith('spmd.py'):
        if IS_WIN:
            return 'mpiexec -localonly -n 4 "' + sys.executable + '" "' + ex + '"'
        return 'mpirun -n 4 "' + sys.executable + '" "' + ex + '"'
    return None


def run_all(nodist=False, nostream=False):
    success = 0
    n = 0
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    for (dirpath, dirnames, filenames) in os.walk(exdir):
        for script in filenames:
            if any(script.endswith(x) for x in ['spmd.py',
                                                'streaming.py',
                                                'stream.py',
                                                'batch.py']):
                n += 1
                logfn = jp(logdir, script.replace('.py', '.res'))
                with open(logfn, 'w') as logfile:
                    print('\n##### ' + jp(dirpath, script))
                    execute_string = get_exe_cmd(jp(dirpath, script), nodist, nostream)
                    if execute_string:
                        os.chdir(dirpath)
                        proc = subprocess.Popen(
                            execute_string if IS_WIN else ['/bin/bash',
                                                           '-c',
                                                           execute_string],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            shell=False
                        )
                        out = proc.communicate()[0]
                        logfile.write(out.decode('ascii'))
                        if proc.returncode:
                            print(out)
                            print(
                                strftime("%H:%M:%S", gmtime()) + '\tFAILED'
                                '\t' + script + '\twith errno\t' + str(proc.returncode)
                            )
                        else:
                            success += 1
                            print(strftime("%H:%M:%S", gmtime()) + '\tPASSED\t' + script)
                    else:
                        success += 1
                        print(strftime("%H:%M:%S", gmtime()) + '\tSKIPPED\t' + script)

    if success != n:
        print('{}/{} examples passed/skipped, {} failed'.format(success, n, n - success))
        print('Error(s) occured. Logs can be found in ' + logdir)
        return 4711
    print('{}/{} examples passed/skipped'.format(success, n))
    return 0


if __name__ == '__main__':
    sys.exit(run_all('nodist' in sys.argv or not __has_dist__, 'nostream' in sys.argv))
