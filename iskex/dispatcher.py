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

from daal4py.sklearn import patch_sklearn as patch_sklearn_orig
from daal4py.sklearn import unpatch_sklearn as unpatch_sklearn_orig
from daal4py.sklearn import sklearn_patch_names as sklearn_patch_names_orig
from daal4py.sklearn import sklearn_patch_map as sklearn_patch_map_orig


def patch_sklearn(name=None, verbose=True):
    patch_sklearn_orig(name, verbose, deprecation=False)


def unpatch_sklearn(name=None):
    unpatch_sklearn_orig(name)


def sklearn_patch_names():
    sklearn_patch_names_orig()


def sklearn_patch_map():
    sklearn_patch_map_orig()
