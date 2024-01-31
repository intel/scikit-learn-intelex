# ==============================================================================
# Copyright 2014 Intel Corporation
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

import argparse
import os
import struct
import subprocess
import sys
from collections import defaultdict
from os.path import join as jp
from time import gmtime, strftime

from daal4py import __has_dist__
from daal4py.sklearn._utils import get_daal_version

print("Starting examples validation")
# First item is major version - 2021,
# second is minor+patch - 0110,
# third item is status - B
print("DAAL version:", get_daal_version())

runner_path = os.path.realpath(__file__)
runner_dir = os.path.dirname(runner_path)
examples_rootdir = jp(
    os.path.dirname(os.path.abspath(os.path.join(runner_path, os.pardir))), "examples"
)
tests_rootdir = jp(
    os.path.dirname(os.path.abspath(os.path.join(runner_path, os.pardir))), "tests"
)

IS_WIN = False
IS_MAC = False
IS_LIN = False
system_os = "not_supported"
if "linux" in sys.platform:
    IS_LIN = True
    system_os = "lnx"
elif sys.platform == "darwin":
    IS_MAC = True
    system_os = "mac"
elif sys.platform in ["win32", "cygwin"]:
    IS_WIN = True
    system_os = "win"
else:
    assert False, sys.platform + " not supported"

assert 8 * struct.calcsize("P") in [32, 64]

if 8 * struct.calcsize("P") == 32:
    logdir = jp(runner_dir, "_results", "ia32")
else:
    logdir = jp(runner_dir, "_results", "intel64")

ex_log_dirs = [
    (jp(examples_rootdir, "daal4py"), jp(logdir, "daal4py")),
    (jp(examples_rootdir, "sklearnex"), jp(logdir, "sklearnex")),
    (jp(tests_rootdir, "daal4py"), jp(logdir, "daal4py")),
]

available_devices = []

try:
    from daal4py.oneapi import sycl_context

    sycl_extention_available = True
except ModuleNotFoundError:
    sycl_extention_available = False
print("Sycl extensions available: {}".format(sycl_extention_available))

if sycl_extention_available:
    try:
        with sycl_context("gpu"):
            gpu_available = True
            available_devices.append("gpu")
    except RuntimeError:
        gpu_available = False
    available_devices.append("cpu")
    # validate that host and cpu devices avaialbe for logging reasons. Examples and
    # vaidaton logic assumes that host and cpu devices are always available
    print("Sycl gpu device: {}".format(gpu_available))


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


# Examples timeout in seconds
execution_timeout = 120

req_version = defaultdict(lambda: (2019, "P", 0))
req_version["decision_forest_classification_hist.py"] = (2023, "P", 100)
req_version["decision_forest_classification_default_dense.py"] = (2023, "P", 100)
req_version["decision_forest_classification_traverse.py"] = (2023, "P", 100)
req_version["basic_statistics_spmd.py"] = (2023, "P", 100)
# Temporary disabling due to sporadict timeout on PVC
req_version["kmeans_spmd.py"] = (2024, "P", 201)
req_version["knn_bf_classification_spmd.py"] = (2023, "P", 100)
req_version["knn_bf_regression_spmd.py"] = (2023, "P", 100)
req_version["linear_regression_spmd.py"] = (2023, "P", 100)
req_version["logistic_regression_spmd.py"] = (2024, "P", 100)

req_device = defaultdict(lambda: [])
req_device["basic_statistics_spmd.py"] = ["gpu"]
req_device["dbscan_spmd.py"] = ["gpu"]
req_device["kmeans_spmd.py"] = ["gpu"]
req_device["knn_bf_classification_dpnp.py"] = ["gpu"]
req_device["knn_bf_classification_spmd.py"] = ["gpu"]
req_device["knn_bf_regression_spmd.py"] = ["gpu"]
req_device["linear_regression_spmd.py"] = ["gpu"]
req_device["logistic_regression_spmd.py"] = ["gpu"]
req_device["pca_spmd.py"] = ["gpu"]
req_device["random_forest_classifier_dpctl.py"] = ["gpu"]
req_device["random_forest_classifier_spmd.py"] = ["gpu"]
req_device["random_forest_regressor_dpnp.py"] = ["gpu"]
req_device["random_forest_regressor_spmd.py"] = ["gpu"]
req_device["sycl/gradient_boosted_regression.py"] = ["gpu"]

req_library = defaultdict(lambda: [])
req_library["basic_statistics_spmd.py"] = ["dpctl", "mpi4py"]
req_library["dbscan_spmd.py"] = ["dpctl", "mpi4py"]
req_library["basic_statistics_spmd.py"] = ["dpctl", "mpi4py"]
req_library["kmeans_spmd.py"] = ["dpctl", "mpi4py"]
req_library["knn_bf_classification_dpnp.py"] = ["dpctl", "dpnp"]
req_library["knn_bf_classification_spmd.py"] = ["dpctl", "mpi4py"]
req_library["knn_bf_regression_spmd.py"] = ["dpctl", "mpi4py"]
req_library["linear_regression_spmd.py"] = ["dpctl", "mpi4py"]
req_library["logistic_regression_spmd.py"] = ["dpctl", "mpi4py"]
req_library["pca_spmd.py"] = ["dpctl", "mpi4py"]
req_library["random_forest_classifier_dpctl.py"] = ["dpctl"]
req_library["random_forest_classifier_spmd.py"] = ["dpctl", "mpi4py"]
req_library["random_forest_regressor_dpnp.py"] = ["dpnp"]
req_library["random_forest_regressor_spmd.py"] = ["dpctl", "dpnp", "mpi4py"]

req_os = defaultdict(lambda: [])

skiped_files = []


def get_exe_cmd(ex, args):
    if os.path.dirname(ex).endswith("sycl"):
        if not sycl_extention_available:
            return None
        if not check_version(
            req_version["sycl/" + os.path.basename(ex)], get_daal_version()
        ):
            return None
        if not check_device(
            req_device["sycl/" + os.path.basename(ex)], available_devices
        ):
            return None
        if not check_os(req_os["sycl/" + os.path.basename(ex)], system_os):
            return None

    if os.path.dirname(ex).endswith("daal4py") or os.path.dirname(ex).endswith("mb"):
        if args.nodaal4py:
            return None
        if not check_version(req_version[os.path.basename(ex)], get_daal_version()):
            return None
        if not check_library(req_library[os.path.basename(ex)]):
            return None

    if os.path.dirname(ex).endswith("sklearnex"):
        if args.nosklearnex:
            return None
        if not check_device(req_device[os.path.basename(ex)], available_devices):
            return None
        if not check_version(req_version[os.path.basename(ex)], get_daal_version()):
            return None
        if not check_library(req_library[os.path.basename(ex)]):
            return None
    if not args.nodist and ex.endswith("spmd.py"):
        if IS_WIN:
            return 'mpiexec -localonly -n 4 "' + sys.executable + '" "' + ex + '"'
        return 'mpirun -n 4 "' + sys.executable + '" "' + ex + '"'
    else:
        return '"' + sys.executable + '" "' + ex + '"'


def run(exdir, logdir, args):
    success = 0
    n = 0
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    for dirpath, dirnames, filenames in os.walk(exdir):
        for script in filenames:
            if script.endswith(".py") and script not in ["__init__.py"]:
                n += 1
                if script in skiped_files:
                    success += 1
                    print("\n##### " + jp(dirpath, script))
                    print(
                        strftime("%H:%M:%S", gmtime())
                        + "\tKNOWN BUG IN EXAMPLES\t"
                        + script
                    )
                else:
                    logfn = jp(logdir, script.replace(".py", ".res"))
                    with open(logfn, "w") as logfile:
                        print("\n##### " + jp(dirpath, script))
                        execute_string = get_exe_cmd(jp(dirpath, script), args)
                        if execute_string:
                            os.chdir(dirpath)
                            try:
                                proc = subprocess.Popen(
                                    (
                                        execute_string
                                        if IS_WIN
                                        else ["/bin/bash", "-c", execute_string]
                                    ),
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    shell=False,
                                )
                                out = proc.communicate(timeout=execution_timeout)[0]
                            except subprocess.TimeoutExpired:
                                proc.kill()
                                out = proc.communicate()[0]
                                print("Process has timed out: " + str(execute_string))
                            logfile.write(out.decode("ascii"))
                            if proc.returncode:
                                print(out)
                                print(
                                    strftime("%H:%M:%S", gmtime()) + "\tFAILED"
                                    "\t" + script + "\twith errno"
                                    "\t" + str(proc.returncode)
                                )
                            else:
                                success += 1
                                print(
                                    strftime("%H:%M:%S", gmtime()) + "\t"
                                    "PASSED\t" + script
                                )
                        else:
                            success += 1
                            print(strftime("%H:%M:%S", gmtime()) + "\tSKIPPED\t" + script)
    return success, n


def run_all(args):
    if args.assert_gpu and "gpu" not in available_devices:
        raise RuntimeError("GPU device not available or not detected")

    success = 0
    num = 0
    for edir, ldir in ex_log_dirs:
        s, n = run(edir, ldir, args)
        success += s
        num += n
    if success != num:
        print(
            "{}/{} examples passed/skipped, "
            "{} failed".format(success, num, num - success)
        )
        print("Error(s) occured. Logs can be found in " + logdir)
        return 4711
    print("{}/{} examples passed/skipped".format(success, num))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nodist",
        action="store_true",
        default=False,
        help="Skip examples with distributed execution",
    )
    parser.add_argument(
        "--nostream",
        action="store_true",
        default=False,
        help="Skip examples with data streaming",
    )
    parser.add_argument(
        "--nosklearnex",
        action="store_true",
        default=False,
        help="Skip sklearnex examples",
    )
    parser.add_argument(
        "--nodaal4py",
        action="store_true",
        default=False,
        help="Skip daal4py examples",
    )
    parser.add_argument(
        "--assert-gpu",
        action="store_true",
        default=False,
        help="Assert a GPU is available",
    )
    sys.exit(run_all(parser.parse_args()))
