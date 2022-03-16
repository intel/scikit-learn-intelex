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
from functools import lru_cache
from daal4py.sklearn._utils import daal_check_version, sklearn_check_version


def _is_new_patching_available():
    return os.environ.get('OFF_ONEDAL_IFACE') is None \
        and daal_check_version((2021, 'P', 300))


@lru_cache(maxsize=None)
def get_patch_map():
    from daal4py.sklearn.monkeypatch.dispatcher import _get_map_of_algorithms
    mapping = _get_map_of_algorithms().copy()

    if _is_new_patching_available():
        # Classes for patching

        from ._config import set_config as set_config_sklearnex
        from ._config import get_config as get_config_sklearnex
        from ._config import config_context as config_context_sklearnex

        from .svm import SVR as SVR_sklearnex
        from .svm import SVC as SVC_sklearnex
        from .svm import NuSVR as NuSVR_sklearnex
        from .svm import NuSVC as NuSVC_sklearnex

        from .neighbors import KNeighborsClassifier as KNeighborsClassifier_sklearnex
        from .neighbors import KNeighborsRegressor as KNeighborsRegressor_sklearnex
        from .neighbors import NearestNeighbors as NearestNeighbors_sklearnex

        # Scikit-learn* modules

        import sklearn as base_module
        import sklearn.svm as svm_module
        import sklearn.neighbors as neighbors_module

        # Patch for mapping
        # Algorithms
        # SVM
        mapping.pop('svm')
        mapping.pop('svc')
        mapping['svr'] = [[(svm_module, 'SVR', SVR_sklearnex), None]]
        mapping['svc'] = [[(svm_module, 'SVC', SVC_sklearnex), None]]
        mapping['nusvr'] = [[(svm_module, 'NuSVR', NuSVR_sklearnex), None]]
        mapping['nusvc'] = [[(svm_module, 'NuSVC', NuSVC_sklearnex), None]]

        # kNN
        mapping.pop('knn_classifier')
        mapping.pop('kneighborsclassifier')
        mapping.pop('knn_regressor')
        mapping.pop('kneighborsregressor')
        mapping.pop('nearest_neighbors')
        mapping.pop('nearestneighbors')
        mapping['knn_classifier'] = [[(neighbors_module,
                                       'KNeighborsClassifier',
                                       KNeighborsClassifier_sklearnex), None]]
        mapping['knn_regressor'] = [[(neighbors_module,
                                      'KNeighborsRegressor',
                                      KNeighborsRegressor_sklearnex), None]]
        mapping['nearest_neighbors'] = [[(neighbors_module,
                                          'NearestNeighbors',
                                          NearestNeighbors_sklearnex), None]]
        mapping['kneighborsclassifier'] = mapping['knn_classifier']
        mapping['kneighborsregressor'] = mapping['knn_regressor']
        mapping['nearestneighbors'] = mapping['nearest_neighbors']

        # Configs
        mapping['set_config'] = [[(base_module,
                                   'set_config',
                                   set_config_sklearnex), None]]
        mapping['get_config'] = [[(base_module,
                                   'get_config',
                                   get_config_sklearnex), None]]
        mapping['config_context'] = [[(base_module,
                                      'config_context',
                                       config_context_sklearnex), None]]
    return mapping


def get_patch_names():
    return list(get_patch_map().keys())


def patch_sklearn(name=None, verbose=True, global_patch=False):
    if not sklearn_check_version('0.22'):
        raise NotImplementedError("Intel(R) Extension for Scikit-learn* patches apply "
                                  "for scikit-learn >= 0.22 only ...")

    if global_patch:
        from sklearnex.glob.dispatcher import patch_sklearn_global
        patch_sklearn_global(name, verbose)

    from daal4py.sklearn import patch_sklearn as patch_sklearn_orig

    if _is_new_patching_available():
        for config in ['set_config', 'get_config', 'config_context']:
            patch_sklearn_orig(config, verbose=False, deprecation=False,
                               get_map=get_patch_map)
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


def unpatch_sklearn(name=None, global_unpatch=False):
    if global_unpatch:
        from sklearnex.glob.dispatcher import unpatch_sklearn_global
        unpatch_sklearn_global()
    from daal4py.sklearn import unpatch_sklearn as unpatch_sklearn_orig

    if isinstance(name, list):
        for algorithm in name:
            unpatch_sklearn_orig(algorithm, get_map=get_patch_map)
    else:
        if _is_new_patching_available():
            for config in ['set_config', 'get_config', 'config_context']:
                unpatch_sklearn_orig(config, get_map=get_patch_map)
        unpatch_sklearn_orig(name, get_map=get_patch_map)
