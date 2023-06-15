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

from daal4py.sklearn._utils import daal_check_version
import logging

if daal_check_version((2023, 'P', 200)):
    import numpy as np
    from scipy.sparse import issparse

    from ._common import BaseKMeans
    from ..._device_offload import dispatch, wrap_output_data

    from onedal.kmeans import KMeans as onedal_KMeans
    from sklearn.cluster import KMeans as sklearn_KMeans

    from daal4py.sklearn._utils import (
        sklearn_check_version,
        PatchingConditionsChain)

    from sklearn.utils.validation import (
        check_is_fitted,
        _num_samples,
        _deprecate_positional_args)

    from sklearn.utils._openmp_helpers import _openmp_effective_n_threads

    class KMeans(sklearn_KMeans, BaseKMeans):
        __doc__ = sklearn_KMeans.__doc__
        n_iter_, inertia_ = None, None
        labels_, cluster_centers_ = None, None

        if sklearn_check_version('1.2'):
            _parameter_constraints: dict = {
                **sklearn_KMeans._parameter_constraints}

            @_deprecate_positional_args
            def __init__(
                self,
                n_clusters=8,
                *,
                init='k-means++',
                n_init='auto' if sklearn_check_version('1.4') else 'warn',
                max_iter=300,
                tol=1e-4,
                verbose=0,
                random_state=None,
                copy_x=True,
                algorithm='lloyd',
            ):
                super().__init__(
                    n_clusters=n_clusters,
                    init=init,
                    max_iter=max_iter,
                    tol=tol,
                    n_init=n_init,
                    verbose=verbose,
                    random_state=random_state,
                    copy_x=copy_x,
                    algorithm=algorithm,
                )
        elif sklearn_check_version('1.0'):
            @_deprecate_positional_args
            def __init__(
                self,
                n_clusters=8,
                *,
                init='k-means++',
                n_init=10,
                max_iter=300,
                tol=1e-4,
                verbose=0,
                random_state=None,
                copy_x=True,
                algorithm='auto',
            ):
                super().__init__(
                    n_clusters=n_clusters,
                    init=init,
                    max_iter=max_iter,
                    tol=tol,
                    n_init=n_init,
                    verbose=verbose,
                    random_state=random_state,
                    copy_x=copy_x,
                    algorithm=algorithm,
                )
        else:
            @_deprecate_positional_args
            def __init__(
                self,
                n_clusters=8,
                *,
                init='k-means++',
                n_init=10,
                max_iter=300,
                tol=1e-4,
                precompute_distances='deprecated',
                verbose=0,
                random_state=None,
                copy_x=True,
                n_jobs='deprecated',
                algorithm='auto',
            ):
                super().__init__(
                    n_clusters=n_clusters,
                    init=init,
                    max_iter=max_iter,
                    tol=tol,
                    precompute_distances=precompute_distances,
                    n_init=n_init,
                    verbose=verbose,
                    random_state=random_state,
                    copy_x=copy_x,
                    n_jobs=n_jobs,
                    algorithm=algorithm,
                )

        def _initialize_onedal_estimator(self):
            onedal_params = {
                'n_clusters': self.n_clusters,
                'init': self.init,
                'max_iter': self.max_iter,
                'tol': self.tol,
                'n_init': self.n_init,
                'verbose': self.verbose,
                'random_state': self.random_state,
            }

            print("!"*16, onedal_params)

            self._onedal_estimator = onedal_KMeans(**onedal_params)

        def _onedal_fit_supported(self, method_name, *data):
            assert method_name == 'fit'
            assert len(data) == 3

            X, y, sample_weight = data

            class_name = self.__class__.__name__
            patching_status = PatchingConditionsChain(
                f'sklearn.cluster.{class_name}.fit')

            supported_algs = ["auto", "full", "lloyd"]

            dal_ready = patching_status.and_conditions([
                (self.algorithm in supported_algs, 'Only lloyd algorithm is supported.'),
                (not issparse(self.init), 'Sparse init values are not supported'),
                (sample_weight is None, 'Sample weight is not None.'),
                (not issparse(X), 'Sparse input is not supported.'),
            ])

            return patching_status.get_status(logs=True)

        def fit(self, X, y=None, sample_weight=None):
            print("!" * 80)
            """Compute k-means clustering.

            Parameters
            ----------
            X : array-like or sparse matrix, shape=(n_samples, n_features)
                Training instances to cluster. It must be noted that the data
                will be converted to C ordering, which will cause a memory
                copy if the given data is not C-contiguous.

            y : Ignored
                not used, present here for API consistency by convention.

            sample_weight : array-like, shape (n_samples,), optional
                The weights for each observation in X. If None, all observations
                are assigned equal weight (default: None)

            """

            if sklearn_check_version('1.0'):
                self._check_feature_names(X, reset=True)
            if sklearn_check_version("1.2"):
                self._validate_params()

            dispatch(self, 'fit', {
                'onedal': self.__class__._onedal_fit,
                'sklearn': sklearn_KMeans.fit,
            }, X, y, sample_weight)

            return self

        def _onedal_fit(self, X, y, sample_weight, queue = None):
            assert sample_weight is None

            self._n_threads = _openmp_effective_n_threads()

            self._initialize_onedal_estimator()
            self._onedal_estimator.fit(X, queue = queue)

            self._save_attributes()

        def _onedal_predict_supported(self, method_name, *data):
            assert method_name == 'predict'
            assert len(data) == 1

            X = data[0]

            class_name = self.__class__.__name__
            patching_status = PatchingConditionsChain(
                f'sklearn.cluster.{class_name}.predict')

            supported_algs = ["auto", "full", "lloyd"]
            dal_ready = patching_status.and_conditions([
                (self.algorithm in supported_algs, 'Only lloyd algorithm is supported.'),
                (not issparse(X), 'Sparse input is not supported.')
            ])

            return patching_status.get_status(logs=True)

        @wrap_output_data
        def predict(self, X):
            """Compute k-means clustering.

            Parameters
            ----------
            X : array-like or sparse matrix, shape=(n_samples, n_features)
                Training instances to cluster. It must be noted that the data
                will be converted to C ordering, which will cause a memory
                copy if the given data is not C-contiguous.

            y : Ignored
                not used, present here for API consistency by convention.

            sample_weight : array-like, shape (n_samples,), optional
                The weights for each observation in X. If None, all observations
                are assigned equal weight (default: None)

            """

            if sklearn_check_version('1.0'):
                self._check_feature_names(X, reset=True)
            if sklearn_check_version("1.2"):
                self._validate_params()

            dispatch(self, 'fit', {
                'onedal': self.__class__._onedal_predict,
                'sklearn': sklearn_KMeans.predict,
            }, X)

            return self

        def _onedal_predict(self, X, queue=None):
            X = self._validate_data(X, accept_sparse=False, reset=False)
            if not hasattr(self, '_onedal_estimator'):
                self._initialize_onedal_estimator()
                self._onedal_estimator.cluster_centers_ = self.cluster_centers_

            return self._onedal_estimator.predict(X, queue=queue)

        def _onedal_supported(self, method_name, *data):
            if method_name == 'fit':
                return self._onedal_fit_supported(method_name, *data)
            if method_name == 'predict':
                return self._onedal_predict_supported(method_name, *data)
            raise RuntimeError(
                f'Unknown method {method_name} in {self.__class__.__name__}')

        def _onedal_gpu_supported(self, method_name, *data):
            return self._onedal_supported(method_name, *data)

        def _onedal_cpu_supported(self, method_name, *data):
            return self._onedal_supported(method_name, *data)

        @wrap_output_data
        def fit_transform(self, X, y=None, sample_weight=None):
            """Compute clustering and transform X to cluster-distance space.

            Equivalent to fit(X).transform(X), but more efficiently implemented.

            Parameters
            ----------
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
                New data to transform.

            y : Ignored
                Not used, present here for API consistency by convention.

            sample_weight : array-like of shape (n_samples,), default=None
                The weights for each observation in X. If None, all observations
                are assigned equal weight.

            Returns
            -------
            X_new : ndarray of shape (n_samples, n_clusters)
                X transformed in the new space.
            """
            return self.fit(X, sample_weight=sample_weight).transform(X)

        @wrap_output_data
        def transform(self, X):
            """Transform X to a cluster-distance space.

            In the new space, each dimension is the distance to the cluster
            centers. Note that even if X is sparse, the array returned by
            `transform` will typically be dense.

            Parameters
            ----------
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
                New data to transform.

            Returns
            -------
            X_new : ndarray of shape (n_samples, n_clusters)
                X transformed in the new space.
            """
            check_is_fitted(self)

            X = self._check_test_data(X)
            return None

else:
    from daal4py.sklearn.cluster import KMeans
    logging.warning('Preview KMeans requires oneDAL version >= 2023.2 '
                    'but it was not found')
