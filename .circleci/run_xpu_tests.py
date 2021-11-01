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


def get_context(device):
    from sklearnex._config import config_context
    return config_context(target_offload=device, allow_fallback_to_host=True)


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
        choices=['host', 'cpu', 'gpu']
    )
    parser.add_argument(
        '--deselect',
        help='The list of deselect commands passed directly to pytest',
        action='append',
        required=True
    )
    args = parser.parse_args()

    deselected_tests = [
        element for test in args.deselect
        for element in ('--deselect', test)
    ]

    pytest_params = [
        "-ra", "--disable-warnings"
    ]

    if not args.quiet:
        pytest_params.append("-q")

    with get_context(args.device):
        pytest.main(
            pytest_params + ["--pyargs", "sklearn"] + deselected_tests
        )
