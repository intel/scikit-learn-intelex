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
import os
from distutils.version import LooseVersion
from functools import lru_cache
from daal4py.sklearn._utils import daal_check_version

# Classes for patching
if os.environ.get('OFF_ONEDAL_IFACE') is None and daal_check_version((2021, 'P', 300)):
    from .svm import SVR as SVR_sklearnex
    from .svm import SVC as SVC_sklearnex
    from .svm import NuSVR as NuSVR_sklearnex
    from .svm import NuSVC as NuSVC_sklearnex

# Scikit-learn* modules
import sklearn.svm as svm_module


@lru_cache(maxsize=None)
def get_patch_map():
    from daal4py.sklearn.monkeypatch.dispatcher import _get_map_of_algorithms
    mapping = _get_map_of_algorithms().copy()

    if os.environ.get('OFF_ONEDAL_IFACE') is None and daal_check_version((2021, 'P', 300)):
        mapping.pop('svm')
        mapping.pop('svc')
        mapping['svr'] = [[(svm_module, 'SVR', SVR_sklearnex), None]]
        mapping['svc'] = [[(svm_module, 'SVC', SVC_sklearnex), None]]
        mapping['nusvr'] = [[(svm_module, 'NuSVR', NuSVR_sklearnex), None]]
        mapping['nusvc'] = [[(svm_module, 'NuSVC', NuSVC_sklearnex), None]]
    return mapping


def get_patch_names():
    return list(get_patch_map().keys())


def patch_sklearn(name=None, verbose=True):
    from sklearn import __version__ as sklearn_version
    if LooseVersion(sklearn_version) < LooseVersion("0.22.0"):
        raise NotImplementedError("Intel(R) Extension for Scikit-learn* patches apply "
                                  "for scikit-learn >= 0.22.0 only ...")

    from daal4py.sklearn import patch_sklearn as patch_sklearn_orig
    if isinstance(name, list):
        for algorithm in name:
            patch_sklearn_orig(algorithm, verbose=False, deprecation=False,
                               get_map=get_patch_map)
    else:
        patch_sklearn_orig(name, verbose=False, deprecation=False,
                           get_map=get_patch_map)

    if verbose and sys.stderr is not None:
        sys.stderr.write(
            "Intel(R) Extension for Scikit-learn* enabled "
            "(https://github.com/intel/scikit-learn-intelex)\n")


def unpatch_sklearn(name=None):
    from daal4py.sklearn import unpatch_sklearn as unpatch_sklearn_orig
    if isinstance(name, list):
        for algorithm in name:
            unpatch_sklearn_orig(algorithm, get_map=get_patch_map)
    else:
        unpatch_sklearn_orig(name, get_map=get_patch_map)
