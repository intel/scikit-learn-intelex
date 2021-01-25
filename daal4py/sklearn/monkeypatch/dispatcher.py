#===============================================================================
# Copyright 2014-2021 Intel Corporation
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

from daal4py.sklearn._utils import daal_check_version
from ..neighbors import KNeighborsRegressor as KNeighborsRegressor_daal4py
from ..neighbors import NearestNeighbors as NearestNeighbors_daal4py
from ..neighbors import KNeighborsClassifier as KNeighborsClassifier_daal4py
from ..model_selection import _daal_train_test_split
from ..utils.validation import _daal_assert_all_finite
from ..svm.svm import SVC as SVC_daal4py
from ..ensemble._forest import RandomForestClassifier as RandomForestClassifier_daal4py
from ..ensemble._forest import RandomForestRegressor as RandomForestRegressor_daal4py
from ..metrics import _daal_roc_auc_score
from ..cluster.k_means import KMeans as KMeans_daal4py
from ..cluster.dbscan import DBSCAN as DBSCAN_daal4py
from ..linear_model.coordinate_descent import Lasso as Lasso_daal4py
from ..linear_model.coordinate_descent import ElasticNet as ElasticNet_daal4py
from ..linear_model.linear import LinearRegression as LinearRegression_daal4py
from ..linear_model.ridge import Ridge as Ridge_daal4py
from ..decomposition._pca import PCA as PCA_daal4py
from ..manifold import TSNE as TSNE_daal4py
from ..linear_model.logistic_path import LogisticRegression as LogisticRegression_daal4py
from sklearn import model_selection
from sklearn import metrics
from sklearn.utils import validation
from sklearn.metrics import pairwise
import sklearn.neighbors as neighbors_module
import sklearn.decomposition as decomposition_module
import sklearn.linear_model as linear_model_module
import sklearn.manifold as manifold_module
import sys
from sklearn import __version__ as sklearn_version
from distutils.version import LooseVersion
from functools import lru_cache

import sklearn.cluster as cluster_module
import sklearn.ensemble as ensemble_module
import sklearn.svm as svm_module

if LooseVersion(sklearn_version) >= LooseVersion("0.22"):
    import sklearn.linear_model._logistic as logistic_module
    _patched_log_reg_path_func_name = '_logistic_regression_path'
    from ..linear_model._logistic_path_0_22 import _logistic_regression_path as daal_optimized_logistic_path
else:
    import sklearn.linear_model.logistic as logistic_module
    if LooseVersion(sklearn_version) >= LooseVersion("0.21.0"):
        _patched_log_reg_path_func_name = '_logistic_regression_path'
        from ..linear_model._logistic_path_0_21 import _logistic_regression_path as daal_optimized_logistic_path
    else:
        _patched_log_reg_path_func_name = 'logistic_regression_path'
        from ..linear_model._logistic_path_0_21 import logistic_regression_path as daal_optimized_logistic_path


if LooseVersion(sklearn_version) >= LooseVersion("0.22"):
    from ._pairwise_0_22 import daal_pairwise_distances
else:
    from ._pairwise_0_21 import daal_pairwise_distances


@lru_cache(maxsize=None)
def _get_map_of_algorithms():
    mapping = {
        'pca':                      [[(decomposition_module, 'PCA', PCA_daal4py), None]],
        'kmeans':                   [[(cluster_module, 'KMeans', KMeans_daal4py), None]],
        'dbscan':                   [[(cluster_module, 'DBSCAN', DBSCAN_daal4py), None]],
        'distances':                [[(metrics, 'pairwise_distances', daal_pairwise_distances), None]],
        'linear':                   [[(linear_model_module, 'LinearRegression', LinearRegression_daal4py), None]],
        'ridge':                    [[(linear_model_module, 'Ridge', Ridge_daal4py), None]],
        'elasticnet':               [[(linear_model_module, 'ElasticNet', ElasticNet_daal4py), None]],
        'lasso':                    [[(linear_model_module, 'Lasso', Lasso_daal4py), None]],
        'svm':                      [[(svm_module, 'SVC', SVC_daal4py), None]],
        'logistic':                 [[(logistic_module, _patched_log_reg_path_func_name, daal_optimized_logistic_path), None]],
        'knn_classifier':           [[(neighbors_module, 'KNeighborsClassifier', KNeighborsClassifier_daal4py), None]],
        'nearest_neighbors':        [[(neighbors_module, 'NearestNeighbors', NearestNeighbors_daal4py), None]],
        'knn_regressor':            [[(neighbors_module, 'KNeighborsRegressor', KNeighborsRegressor_daal4py), None]],
        'random_forest_classifier': [[(ensemble_module, 'RandomForestClassifier', RandomForestClassifier_daal4py), None]],
        'random_forest_regressor':  [[(ensemble_module, 'RandomForestRegressor', RandomForestRegressor_daal4py), None]],
        'train_test_split':         [[(model_selection, 'train_test_split', _daal_train_test_split), None]],
        'fin_check':                [[(validation, '_assert_all_finite', _daal_assert_all_finite), None]],
        'tsne':                     [[(manifold_module, 'TSNE', TSNE_daal4py), None]],
    }
    if daal_check_version((2021,'P', 100)):
        mapping['log_reg'] = [[(linear_model_module, 'LogisticRegression', LogisticRegression_daal4py), None]]
    if daal_check_version((2021,'P', 200)):
        mapping['roc_auc_score'] = [[(metrics, 'roc_auc_score', _daal_roc_auc_score), None]]
    return mapping


def do_patch(name):
    lname = name.lower()
    if lname in _get_map_of_algorithms():
        for descriptor in _get_map_of_algorithms()[lname]:
            which, what, replacer = descriptor[0]
            if descriptor[1] is None:
                descriptor[1] = getattr(which, what, None)
            setattr(which, what, replacer)
    else:
        raise ValueError("Has no patch for: " + name)


def do_unpatch(name):
    lname = name.lower()
    if lname in _get_map_of_algorithms():
        for descriptor in _get_map_of_algorithms()[lname]:
            if descriptor[1] is not None:
                which, what, replacer = descriptor[0]
                setattr(which, what, descriptor[1])
    else:
        raise ValueError("Has no patch for: " + name)


def enable(name=None, verbose=True):
    if LooseVersion(sklearn_version) < LooseVersion("0.21.0"):
        raise NotImplementedError(
            "daal4py patches apply for scikit-learn >= 0.21.0 only ...")
    if name is not None:
        do_patch(name)
    else:
        for key in _get_map_of_algorithms():
            do_patch(key)
    if verbose and sys.stderr is not None:
        sys.stderr.write("Intel(R) oneAPI Data Analytics Library solvers for sklearn enabled: "
                         "https://intelpython.github.io/daal4py/sklearn.html\n")


def disable(name=None):
    if name is not None:
        do_unpatch(name)
    else:
        for key in _get_map_of_algorithms():
            do_unpatch(key)
        _get_map_of_algorithms.cache_clear()


def _patch_names():
    return list(_get_map_of_algorithms().keys())
