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

# Other imports
import sys
from distutils.version import LooseVersion
from functools import lru_cache
from daal4py.sklearn._utils import daal_check_version

# Classes for patching
if daal_check_version((2021, 'P', 300)):
    from .svm import SVR as SVR_sklearnex
    from .svm import SVC as SVC_sklearnex

# Scikit-learn* modules
import sklearn.svm as svm_module


@lru_cache(maxsize=None)
def _get_map_of_algorithms_sklearnex():
    mapping = {}

    if daal_check_version((2021, 'P', 300)):
        mapping['svr'] = [[(svm_module, 'SVR', SVR_sklearnex), None]]
        mapping['svc'] = [[(svm_module, 'SVC', SVC_sklearnex), None]]

    return mapping


@lru_cache(maxsize=None)
def _get_d4p_map_without_duplicates():
    from daal4py.sklearn import sklearn_patch_map
    sklearnex_classes = [c[0][0][1] for c in _get_map_of_algorithms_sklearnex().values()]
    new_map = {}
    for key, value in sklearn_patch_map().items():
        if value[0][0][1] not in sklearnex_classes:
            new_map.update({key: value})
    return new_map


def patch_sklearn(name=None, verbose=True):
    from sklearn import __version__ as sklearn_version
    if LooseVersion(sklearn_version) < LooseVersion("0.22.0"):
        raise NotImplementedError("Intel(R) Extension for Scikit-learn* patches apply "
                                  "for scikit-learn >= 0.22.0 only ...")

    from daal4py.sklearn import patch_sklearn as patch_sklearn_orig
    if isinstance(name, list):
        for algorithm in name:
            patch_sklearn_orig(algorithm, verbose=False, deprecation=False,
                               get_map=_get_d4p_map_without_duplicates, d4p_only=False)
            patch_sklearn_orig(algorithm, verbose=False, deprecation=False,
                               get_map=_get_map_of_algorithms_sklearnex, d4p_only=False)
    else:
        patch_sklearn_orig(name, verbose=False, deprecation=False,
                           get_map=_get_d4p_map_without_duplicates, d4p_only=False)
        patch_sklearn_orig(name, verbose=False, deprecation=False,
                           get_map=_get_map_of_algorithms_sklearnex, d4p_only=False)

    if verbose and sys.stderr is not None:
        sys.stderr.write(
            "Intel(R) Extension for Scikit-learn* enabled "
            "(https://github.com/intel/scikit-learn-intelex)\n")


def unpatch_sklearn(name=None):
    from daal4py.sklearn import unpatch_sklearn as unpatch_sklearn_orig
    if isinstance(name, list):
        for algorithm in name:
            unpatch_sklearn_orig(algorithm, get_map=_get_d4p_map_without_duplicates,
                                 d4p_only=False)
            unpatch_sklearn_orig(algorithm, get_map=_get_map_of_algorithms_sklearnex,
                                 d4p_only=False)
    else:
        unpatch_sklearn_orig(name, get_map=_get_d4p_map_without_duplicates,
                             d4p_only=False)
        unpatch_sklearn_orig(name, get_map=_get_map_of_algorithms_sklearnex,
                             d4p_only=False)


def get_patch_names():
    return list(_get_d4p_map_without_duplicates().keys()) + \
        list(_get_map_of_algorithms_sklearnex().keys())
