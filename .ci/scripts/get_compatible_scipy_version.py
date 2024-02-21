#! /usr/bin/env python
# ===============================================================================
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
# ===============================================================================
from sys import exit, stderr
from sys import version_info as python_version

from daal4py.sklearn._utils import sklearn_check_version

if sklearn_check_version("1.4"):
    print("Scipy version is not specified for this sklearn/python version.", file=stderr)
    print("scipy")
elif sklearn_check_version("1.3") or python_version[1] > 11:
    print("scipy==1.11.*")
elif sklearn_check_version("1.2") or python_version[1] > 10:
    print("scipy==1.9.*")
elif sklearn_check_version("1.1"):
    print("scipy==1.8.*")
elif sklearn_check_version("1.0"):
    print("scipy==1.7.*")
elif sklearn_check_version("0.24"):
    # scipy 1.6 is compatible with pandas versions lower than 1.4
    print("scipy==1.6.* pandas==1.3.*")
else:
    print(
        "Scipy version defaults to not specified "
        "for this outdated sklearn/python version.",
        file=stderr,
    )
    print("scipy")
