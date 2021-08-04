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

import subprocess

err_code = subprocess.call(["python", "-m", "sklearnex.globally", "patch_sklearn",
                             "-a", "svc"])
if err_code: raise Exception("Patch error")

from sklearn.svm import SVC
assert SVC.__module__.startswith('daal4py') or SVC.__module__.startswith('sklearnex')

from sklearn.svm import SVR
assert SVR.__module__.startswith('sklearn')

err_code = subprocess.call(["python", "-m", "sklearnex.globally", "unpatch_sklearn"])
if err_code: raise Exception("Unpatch error")

from sklearn.svm import SVC
assert SVC.__module__.startswith('sklearn')

from sklearn.svm import SVR
assert SVR.__module__.startswith('sklearn')
