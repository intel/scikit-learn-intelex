# ==============================================================================
# Copyright 2014 Intel Corporation
# Copyright 2024 Fujitsu Limited
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
import json
import os
import platform as plt
import struct
import subprocess
import sys
from collections import defaultdict
from os.path import join as jp
from time import gmtime, strftime, time

from daal4py import __has_dist__
from daal4py.sklearn._utils import get_daal_version
from onedal._device_offload import dpctl_available

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

arch_dir = plt.machine()
plt_dict = {"x86_64": "intel64", "AMD64": "intel64", "aarch64": "arm"}
arch_dir = plt_dict[arch_dir] if arch_dir in plt_dict else arch_dir
assert 8 * struct.calcsize("P") in [32, 64]
json_log_file_name = "log.json"

if 8 * struct.calcsize("P") == 32:
    logdir = jp(runner_dir, "_results", "ia32")
else:
    logdir = jp(runner_dir, "_results", arch_dir)

ex_log_dirs = [
    (jp(examples_rootdir, "daal4py"), jp(logdir, "daal4py")),
    (jp(examples_rootdir, "sklearnex"), jp(logdir, "sklearnex")),
    (jp(tests_rootdir, "daal4py"), jp(logdir, "daal4py")),
]

available_devices = ["cpu"]

gpu_available = False
if dpctl_available:
    import dpctl

    if dpctl.has_gpu_devices():
        gpu_available = True
        available_devices.append("gpu")

print("GPU device available: {}".format(gpu_available))


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
req_version["kmeans_spmd.py"] = (2024, "P", 200)
req_version["knn_bf_classification_spmd.py"] = (2023, "P", 100)
req_version["knn_bf_regression_spmd.py"] = (2023, "P", 100)
req_version["linear_regression_spmd.py"] = (2023, "P", 100)
req_version["logistic_regression_spmd.py"] = (2024, "P", 400)

req_device = defaultdict(lambda: [])
req_device["basic_statistics_spmd.py"] = ["gpu"]
req_device["covariance_spmd.py"] = ["gpu"]
req_device["dbscan_spmd.py"] = ["gpu"]
req_device["incremental_basic_statistics_dpctl.py"] = ["gpu"]
req_device["incremental_covariance_spmd.py"] = ["gpu"]
req_device["incremental_linear_regression_dpctl.py"] = ["gpu"]
req_device["incremental_pca_dpctl.py"] = ["gpu"]
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

req_library = defaultdict(lambda: [])
req_library["basic_statistics_spmd.py"] = ["dpctl", "mpi4py"]
req_library["covariance_spmd.py"] = ["dpctl", "mpi4py"]
req_library["dbscan_spmd.py"] = ["dpctl", "mpi4py"]
req_library["incremental_basic_statistics_dpctl.py"] = ["dpctl"]
req_library["incremental_covariance_spmd.py"] = ["dpctl", "mpi4py"]
req_library["incremental_linear_regression_dpctl.py"] = ["dpctl"]
req_library["incremental_pca_dpctl.py"] = ["dpctl"]
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
req_os["basic_statistics_spmd.py"] = ["lnx"]
req_os["covariance_spmd.py"] = ["lnx"]
req_os["dbscan_spmd.py"] = ["lnx"]
req_os["kmeans_spmd.py"] = ["lnx"]
req_os["knn_bf_classification_dpnp.py"] = ["lnx"]
req_os["knn_bf_classification_spmd.py"] = ["lnx"]
req_os["knn_bf_regression_spmd.py"] = ["lnx"]
req_os["linear_regression_spmd.py"] = ["lnx"]
req_os["logistic_regression_spmd.py"] = ["lnx"]
req_os["pca_spmd.py"] = ["lnx"]
req_os["random_forest_classifier_dpctl.py"] = ["lnx"]
req_os["random_forest_classifier_spmd.py"] = ["lnx"]
req_os["random_forest_regressor_dpnp.py"] = ["lnx"]
req_os["random_forest_regressor_spmd.py"] = ["lnx"]

skiped_files = []


def append_to_json_file(json_lst, file_name):
    if os.path.isfile(file_name):
        with open(file_name, "r") as json_file:
            current_json_lst = json.load(json_file)
        json_lst = json_lst + current_json_lst
    with open(file_name, "w") as json_file:
        json.dump(json_lst, json_file)


def get_exe_cmd(ex, args):
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
        if not check_os(req_os[os.path.basename(ex)], system_os):
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
    json_logs = list()
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    name_suffix_from_args = ""
    for arg, argv in args.__dict__.items():
        if argv:
            name_suffix_from_args += "_" + arg
    test_type = "unclassified"
    if "daal4py" in logdir:
        test_type = "daal4py"
    elif "sklearnex" in logdir:
        test_type = "sklearnex"

    for dirpath, dirnames, filenames in os.walk(exdir):
        for script in filenames:
            if script.endswith(".py") and script not in ["__init__.py"]:
                n += 1
                json_log_name = script.replace(".py", "") + name_suffix_from_args
                if script in skiped_files:
                    success += 1
                    print("\n##### " + jp(dirpath, script))
                    print(
                        strftime("%H:%M:%S", gmtime())
                        + "\tKNOWN BUG IN EXAMPLES\t"
                        + script
                    )
                    json_log_entry = {
                        "name": json_log_name,
                        "result": "skipped",
                        "reason": "known bug in examples",
                    }
                else:
                    logfn = jp(logdir, script.replace(".py", ".res"))
                    with open(logfn, "w") as logfile:
                        print("\n##### " + jp(dirpath, script))
                        execute_string = get_exe_cmd(jp(dirpath, script), args)
                        if execute_string:
                            os.chdir(dirpath)
                            executed_successfully = False
                            failure_reason = None
                            time_start = time()
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
                                executed_successfully = proc.returncode == os.EX_OK
                                if not executed_successfully:
                                    failure_reason = out.decode()
                            except subprocess.TimeoutExpired:
                                proc.kill()
                                out = proc.communicate()[0]
                                print("Process has timed out: " + str(execute_string))
                                failure_reason = "timeout"
                            time_end = time()
                            logfile.write(out.decode("ascii"))
                            if proc.returncode:
                                print(out.decode())
                                print(
                                    strftime("%H:%M:%S", gmtime()) + "\tFAILED"
                                    "\t" + script + "\twith errno"
                                    "\t" + str(proc.returncode)
                                )
                                failure_reason = out.decode()
                            else:
                                success += 1
                                print(
                                    strftime("%H:%M:%S", gmtime()) + "\t"
                                    "PASSED\t" + script
                                )
                            json_log_entry = {
                                "name": json_log_name,
                                "result": "passed" if executed_successfully else "failed",
                                "duration": time_end - time_start,
                            } | (
                                {"reason": failure_reason}
                                if not executed_successfully
                                else {}
                            )
                        else:
                            success += 1
                            print(strftime("%H:%M:%S", gmtime()) + "\tSKIPPED\t" + script)
                            json_log_entry = {"name": json_log_name, "result": "skipped"}
            json_log_entry = json_log_entry | {"type": test_type}
            json_logs.append(json_log_entry)
    append_to_json_file(json_logs, jp(logdir, json_log_file_name))
    return success, n


def run_all(args):
    if args.assert_gpu and "gpu" not in available_devices:
        raise RuntimeError("GPU device not available or not detected")

    success = 0
    num = 0
    for _, ldir in ex_log_dirs:
        json_log_file_name_ex = jp(ldir, json_log_file_name)
        if os.path.isfile(json_log_file_name_ex):
            os.remove(json_log_file_name_ex)
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
