#===============================================================================
# Copyright 2022 Intel Corporation
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

from abc import ABCMeta

from numbers import Integral

import numpy as np
from ..datatypes import (
    _check_X_y,
    _check_array,
    _column_or_1d,
    _check_n_features,
    _check_classification_targets,
    _num_samples
)

from onedal import _backend

from ..common._policy import _get_policy
from ..datatypes._data_conversion import from_table, to_table

class KMeans(metaclass=ABCMeta):
    def __init__(self, n_clusters=8, *, init='k-means++',
                 n_init=10, max_iter=300, tol=0.0001,
                 verbose=0, random_state=None, copy_x=True,
                 algorithm='auto'):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.algorithm = algorithm

    def _get_onedal_params(self, data):
        return {
            'fptype': 'float' if data.dtype is np.dtype('float32') else 'double',
            'method': 'lloyd_dense',
            'cluster_count': 2,
            'max_iteration_count': 100,
            'accuracy_threshold': 0.1
        }

    def _get_initial_centroids(self):
        centroids = np.asarray([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=np.float64)
        return centroids

    def fit(self, X, queue):
        module = _backend.kmeans.clustering

        policy = _get_policy(queue, X)
        params = self._get_onedal_params(X)
        centroids = self._get_initial_centroids()
        result = module.train(policy, params, *to_table(X, centroids))

        return result
