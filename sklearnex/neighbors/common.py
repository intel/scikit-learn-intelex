#!/usr/bin/env python
#===============================================================================
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
#===============================================================================

from daal4py.sklearn._utils import PatchingConditionsChain
from sklearn.neighbors._base import NeighborsBase as sklearn_NeighborsBase
from sklearn.neighbors._ball_tree import BallTree
from sklearn.neighbors._kd_tree import KDTree
import numpy as np
from scipy import sparse as sp


class KNeighborsDispatchingBase:
    def _onedal_supported(self, device, method_name, *data):
        class_name = self.__class__.__name__
        is_classifier = 'Classifier' in class_name
        is_regressor = 'Regressor' in class_name
        is_unsupervised = not (is_classifier or is_regressor)
        patching_status = PatchingConditionsChain(
            f'sklearn.neighbors.{class_name}.{method_name}')

        if not patching_status.and_condition(
            not isinstance(data[0], (KDTree, BallTree, sklearn_NeighborsBase)),
            f'Input type {type(data[0])} is not supported.'
        ):
            return patching_status.get_status(logs=True)

        if self._fit_method in ['auto', 'ball_tree']:
            condition = self.n_neighbors is not None and \
                self.n_neighbors >= self.n_samples_fit_ // 2
            if self.n_features_in_ > 15 or condition:
                result_method = 'brute'
            else:
                if self.effective_metric_ in ['euclidean']:
                    result_method = 'kd_tree'
                else:
                    result_method = 'brute'
        else:
            result_method = self._fit_method

        p_less_than_one = "p" in self.effective_metric_params_.keys() and \
            self.effective_metric_params_["p"] < 1
        if not patching_status.and_condition(
            not p_less_than_one, '"p" metric parameter is less than 1'
        ):
            return patching_status.get_status(logs=True)

        if not patching_status.and_condition(
            not sp.isspmatrix(data[0]), 'Sparse input is not supported.'
        ):
            return patching_status.get_status(logs=True)

        if not is_unsupervised:
            is_valid_weights = self.weights in ['uniform', "distance"]
            if is_classifier:
                class_count = 1
            is_single_output = False
            y = None
            # To check multioutput, might be overhead
            if len(data) > 1:
                y = np.asarray(data[1])
                if is_classifier:
                    class_count = len(np.unique(y))
            if hasattr(self, '_onedal_estimator'):
                y = self._onedal_estimator._y
            if y is not None and hasattr(y, 'ndim') and hasattr(y, 'shape'):
                is_single_output = y.ndim == 1 or y.ndim == 2 and y.shape[1] == 1

        # TODO: add native support for these metric names
        metrics_map = {
            'manhattan': ['l1', 'cityblock'],
            'euclidean': ['l2']
        }
        for origin, aliases in metrics_map.items():
            if self.effective_metric_ in aliases:
                self.effective_metric_ = origin
                break
        if self.effective_metric_ == 'manhattan':
            self.effective_metric_params_['p'] = 1
        elif self.effective_metric_ == 'euclidean':
            self.effective_metric_params_['p'] = 2

        onedal_brute_metrics = [
            'manhattan', 'minkowski', 'euclidean', 'chebyshev', 'cosine']
        onedal_kdtree_metrics = ['euclidean']
        is_valid_for_brute = result_method == 'brute' and \
            self.effective_metric_ in onedal_brute_metrics
        is_valid_for_kd_tree = result_method == 'kd_tree' and \
            self.effective_metric_ in onedal_kdtree_metrics
        if result_method == 'kd_tree':
            if not patching_status.and_condition(
                device != 'gpu', '"kd_tree" method is not supported on GPU.'
            ):
                return patching_status.get_status(logs=True)

        if not patching_status.and_condition(
            is_valid_for_kd_tree or is_valid_for_brute,
            f'{result_method} with {self.effective_metric_} metric is not supported.'
        ):
            return patching_status.get_status(logs=True)
        if not is_unsupervised:
            if not patching_status.and_conditions([
                (is_single_output, 'Only single output is supported.'),
                (is_valid_weights,
                 f'"{type(self.weights)}" weights type is not supported.')
            ]):
                return patching_status.get_status(logs=True)
        if method_name == 'fit':
            if is_classifier:
                patching_status.and_condition(
                    class_count >= 2, 'One-class case is not supported.'
                )
            return patching_status.get_status(logs=True)
        if method_name in ['predict', 'predict_proba', 'kneighbors']:
            patching_status.and_condition(
                hasattr(self, '_onedal_estimator'), 'oneDAL model was not trained.'
            )
            return patching_status.get_status(logs=True)
        raise RuntimeError(f'Unknown method {method_name} in {class_name}')

    def _onedal_gpu_supported(self, method_name, *data):
        return self._onedal_supported('gpu', method_name, *data)

    def _onedal_cpu_supported(self, method_name, *data):
        return self._onedal_supported('cpu', method_name, *data)
