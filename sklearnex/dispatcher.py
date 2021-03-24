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


def patch_sklearn(name=None, verbose=True):
    from daal4py.sklearn import patch_sklearn as patch_sklearn_orig
    try:
        patch_sklearn_orig(name, verbose, deprecation=False)
    except Exception:
        patch_sklearn_orig(name, verbose)


def unpatch_sklearn(name=None):
    from daal4py.sklearn import unpatch_sklearn as unpatch_sklearn_orig
    unpatch_sklearn_orig(name)


def get_patch_names():
    from daal4py.sklearn import sklearn_patch_names as get_patch_names_orig
    return get_patch_names_orig()
