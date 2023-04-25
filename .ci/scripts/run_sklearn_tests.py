#===============================================================================
# Copyright 2023 Intel Corporation
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

from sklearnex import patch_sklearn
patch_sklearn()

import os
import sys
import argparse
import pytest
import sklearn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--device',
        type=str,
        default='none',
        help='device name',
        choices=['none', 'cpu', 'gpu']
    )
    args = parser.parse_args()

    os.chdir(os.path.dirname(sklearn.__file__))

    if os.environ["SELECTED_TESTS"] == 'all':
        os.environ["SELECTED_TESTS"] = ''

    pytest_args = '--pyargs --durations=100 --durations-min=0.01 ' \
        f'{os.environ["DESELECTED_TESTS"]} {os.environ["SELECTED_TESTS"]}'.split(' ')
    while '' in pytest_args:
        pytest_args.remove('')

    if args.device != 'none':
        with sklearn.config_context(target_offload=args.device):
            return_code = pytest.main(pytest_args)
    else:
        return_code = pytest.main(pytest_args)
    sys.exit(int(return_code))
