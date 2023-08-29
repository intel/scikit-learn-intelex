# ==============================================================================
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
# ==============================================================================

from abc import ABC


def get_cluster_centers(self):
    return self._cluster_centers_


def set_cluster_centers(self, value):
    self._cluster_centers_ = value
    if hasattr(self, "_onedal_estimator"):
        self._onedal_estimator.cluster_centers_ = value


def get_labels(self):
    return self._labels_


def set_labels(self, value):
    self._labels_ = value
    if hasattr(self, "_onedal_estimator"):
        self._onedal_estimator.labels_ = value


def get_inertia(self):
    return self._inertia_


def set_inertia(self, value):
    self._inertia_ = value
    if hasattr(self, "_onedal_estimator"):
        self._onedal_estimator.inertia_ = value


def get_n_iter(self):
    return self._n_iter_


def set_n_iter(self, value):
    self._n_iter_ = value
    if hasattr(self, "_onedal_estimator"):
        self._onedal_estimator.n_iter_ = value


class BaseKMeans(ABC):
    def _save_attributes(self):
        assert hasattr(self, "_onedal_estimator")
        self.n_features_in_ = self._onedal_estimator.n_features_in_
        self.fit_status_ = 0
        self._tol = self._onedal_estimator._tol
        self._n_init = self._onedal_estimator._n_init
        self._n_iter_ = self._onedal_estimator.n_iter_
        self._labels_ = self._onedal_estimator.labels_
        self._inertia_ = self._onedal_estimator.inertia_
        self._algorithm = self._onedal_estimator.algorithm
        self._cluster_centers_ = self._onedal_estimator.cluster_centers_
        self._sparse = False

        self.n_iter_ = property(get_n_iter, set_n_iter)
        self.labels_ = property(get_labels, set_labels)
        self.inertia_ = property(get_labels, set_inertia)
        self.cluster_centers_ = property(get_cluster_centers, set_cluster_centers)

        self._is_in_fit = True
        self.n_iter_ = self._n_iter_
        self.labels_ = self._labels_
        self.inertia_ = self._inertia_
        self.cluster_centers_ = self._cluster_centers_
        self._is_in_fit = False
