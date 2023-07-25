#! /usr/bin/env python
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

import os
from os.path import join as jp


def get_packages_with_tests(pkg_list):
    new_pkg_list = []
    for package in pkg_list:
        new_pkg_list.append(package)

        path = os.path.abspath('./' + package.replace('.', '/'))
        if os.path.isdir(jp(path, 'tests')):
            new_pkg_list.append(package + '.tests')
        if os.path.isdir(jp(path, 'tests', 'utils')):
            new_pkg_list.append(package + '.tests.utils')
    return new_pkg_list
