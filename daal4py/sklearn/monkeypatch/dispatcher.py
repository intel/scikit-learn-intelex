#
#*******************************************************************************
# Copyright 2014-2017 Intel Corporation
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
#******************************************************************************/

import warnings
from sklearn import __version__ as sklearn_version
from distutils.version import LooseVersion
import warnings

import sklearn.cluster as kmeans_module
import sklearn.svm as svm_module
import sklearn.linear_model.logistic as logistic_module
import sklearn.linear_model.base as linear_regression_module
import sklearn.linear_model.ridge as ridge_module
import sklearn.decomposition.pca as pca_module

from sklearn.decomposition.pca import PCA
from sklearn.metrics import pairwise


from .pairwise import daal_pairwise_distances
from .pca import PCA as PCA_daal4py
from .ridge import Ridge as Ridge_daal4py
from .linear import LinearRegression as LinearRegression_daal4py
from .k_means import KMeans as KMeans_daal4py
from .logistic_path import logistic_regression_path as daal_optimized_logistic_path
from .svm import SVC as SVC_daal4py

_mapping = {
    'pca':    [[(pca_module, 'PCA', PCA_daal4py), None]],
    'kmeans':    [[(kmeans_module, 'KMeans', KMeans_daal4py), None]],
    'distances': [[(pairwise, 'pairwise_distances', daal_pairwise_distances), None]],
    'linear':    [[(linear_regression_module, 'LinearRegression', LinearRegression_daal4py)]],
    'ridge':     [[(ridge_module, 'Ridge', Ridge_daal4py)]],
    'svm':       [[(svm_module, 'SVC', SVC_daal4py), None]], 
    'logistic':  [[(logistic_module, 'logistic_regression_path', daal_optimized_logistic_path), None]],
}


def do_patch(name):
    lname = name.lower()
    if lname in _mapping:
        for descriptor in _mapping[lname]:
            which, what, replacer = descriptor[0]
            if descriptor[1] is None:
                descriptor[1] = getattr(which, what, None)
            setattr(which, what, replacer)
    else:
        raise ValueError("Has no patch for: " + name)


def do_unpatch(name):
    lname = name.lower()
    if lname in _mapping:
        for descriptor in _mapping[lname]:
            which, what, replacer = descriptor[0]
            setattr(which, what, descriptor[1])
    else:
        raise ValueError("Has no patch for: " + name)


def enable(name=None):
    if LooseVersion(sklearn_version) < LooseVersion("0.20.0"):
        raise NotImplementedError("daal4sklearn is for scikit-learn 0.20.0 only ...")
    elif LooseVersion(sklearn_version) > LooseVersion("0.20.2"):
        warnings.warn("daal4sklearn {daal4py_version} has only been tested with scikit-learn 0.20.2, found version...")

    if name is not None:
        do_patch(name)
    else:
        for key in _mapping:
            do_patch(key)


def disable(name=None):
    if name is not None:
        do_unpatch(name)
    else:
        for key in _mapping:
            do_unpatch(key)


def _patch_names():
    return list(_mapping.keys())
