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

# coding: utf-8
import argparse
import pytest
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script to run scikit-learn tests with device context manager')
    parser.add_argument(
        '-q', '--quiet',
        help='make pytest less verbose',
        action='store_false')
    parser.add_argument(
        '-d', '--device',
        type=str,
        help='device name',
        choices=['cpu', 'gpu']
    )
    parser.add_argument(
        '--deselect',
        help='The list of deselect commands passed directly to pytest',
        action='append',
        required=False
    )
    parser.add_argument('--no-intel-optimized', default=False, action='store_true',
                        help='Use Scikit-learn without Intel optimizations')
    parser.add_argument('--deselected_yml_file', action='append', type=str)
    parser.add_argument('--absolute', action='store_true')
    parser.add_argument('--reduced', action='store_true')
    parser.add_argument('--public', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    yml_deselected_tests = []
    if args.deselected_yml_file is not None:
        fn = args.deselected_yml_file[0]
        if os.path.exists(fn):
            from deselect_tests import create_pytest_switches
            yml_deselected_tests = create_pytest_switches(fn, args.absolute, args.reduced,
                                                          args.public, args.gpu)

    deselected_tests = []
    if args.deselect is not None:
        deselected_tests = [
            element for test in args.deselect
            for element in ('--deselect', test)
        ]

    yml_deselected_tests = yml_deselected_tests + deselected_tests

    pytest_params = [
        "-ra", "--disable-warnings"
    ]

    if not args.quiet:
        pytest_params.append("-q")

    if not args.no_intel_optimized:
        from sklearnex import patch_sklearn
        patch_sklearn()

    if args.device == "gpu":
        from daal4py.oneapi import sycl_context
        with sycl_context(args.device, host_offload_on_fail=True):
            pytest.main(
                pytest_params + ["--pyargs", "sklearn"] + yml_deselected_tests
            )
    else:
        pytest.main(
            pytest_params + ["--pyargs", "sklearn"] + yml_deselected_tests
        )
