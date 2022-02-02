#!/usr/bin/env python
#===============================================================================
# Copyright 2020 Intel Corporation
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

# coding: utf-8
import argparse
import os.path
from yaml import FullLoader, load as yaml_load
try:
    from packaging.version import Version
except ImportError:
    from distutils.version import LooseVersion as Version
import sklearn
from sklearn import __version__ as sklearn_version
import warnings


def evaluate_cond(cond, v):
    if cond.startswith(">="):
        return Version(v) >= Version(cond[2:])
    if cond.startswith("<="):
        return Version(v) <= Version(cond[2:])
    if cond.startswith("=="):
        return Version(v) == Version(cond[2:])
    if cond.startswith("!="):
        return Version(v) != Version(cond[2:])
    if cond.startswith("<"):
        return Version(v) < Version(cond[1:])
    if cond.startswith(">"):
        return Version(v) > Version(cond[1:])
    warnings.warn(
        'Test selection condition "{0}" should start with '
        '>=, <=, ==, !=, < or > to compare to version of scikit-learn run. '
        'The test will not be deselected'.format(cond))
    return False


def filter_by_version(entry, sk_ver):
    if not entry:
        return None
    t = entry.split(' ')
    if len(t) == 1:
        return entry
    if len(t) != 2:
        return None
    test_name, cond = t
    conds = cond.split(',')
    if all([evaluate_cond(cond, sk_ver) for cond in conds]):
        return test_name
    return None


def create_pytest_switches(filename, absolute, reduced, public, gpu):
    pytest_switches = []
    if os.path.exists(filename):
        with open(filename, 'r') as fh:
            dt = yaml_load(fh, Loader=FullLoader)

        if absolute:
            base_dir = os.path.relpath(
                os.path.dirname(sklearn.__file__),
                os.path.expanduser('~')) + '/'
        else:
            base_dir = ""

        filtered_deselection = [
            filter_by_version(test_name, sklearn_version)
            for test_name in dt.get('deselected_tests', [])]
        if reduced:
            filtered_deselection.extend(
                [filter_by_version(test_name, sklearn_version)
                 for test_name in dt.get('reduced_tests', [])])
        if public:
            filtered_deselection.extend(
                [filter_by_version(test_name, sklearn_version)
                 for test_name in dt.get('public', [])])
        if gpu:
            filtered_deselection.extend(
                [filter_by_version(test_name, sklearn_version)
                 for test_name in dt.get('gpu', [])])
        pytest_switches = []
        for test_name in filtered_deselection:
            if test_name:
                pytest_switches.extend(["--deselect", base_dir + test_name])
    return pytest_switches


if __name__ == '__main__':
    argParser = argparse.ArgumentParser(
        prog="deselect_tests.py",
        description="Produce pytest CLI options to deselect tests specified in yaml file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argParser.add_argument('conf_file', nargs=1, type=str)
    argParser.add_argument('--absolute', action='store_true')
    argParser.add_argument('--reduced', action='store_true')
    argParser.add_argument('--public', action='store_true')
    argParser.add_argument('--gpu', action='store_true')
    args = argParser.parse_args()

    fn = args.conf_file[0]
    if os.path.exists(fn):
        print(" ".join(create_pytest_switches(fn, args.absolute, args.reduced,
                                              args.public, args.gpu)))
