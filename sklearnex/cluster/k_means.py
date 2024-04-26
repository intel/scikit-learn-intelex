# ==============================================================================
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
# ==============================================================================

import logging

from daal4py.sklearn._utils import daal_check_version

if daal_check_version((2023, "P", 200)):
    from abc import ABC

    import numpy as np
    from scipy.sparse import issparse
    from sklearn.cluster import KMeans as sklearn_KMeans
    from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
    from sklearn.utils.validation import (
        _deprecate_positional_args,
        _num_samples,
        check_is_fitted,
    )

    from daal4py.sklearn._device_offload import support_usm_ndarray
    from daal4py.sklearn._n_jobs_support import control_n_jobs
    from daal4py.sklearn._utils import sklearn_check_version
    from onedal.cluster import KMeans as onedal_KMeans

    from .._device_offload import dispatch, wrap_output_data
    from .._utils import PatchingConditionsChain

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

    @control_n_jobs(decorated_methods=["fit", "predict"])
    class KMeans(sklearn_KMeans, BaseKMeans):
        __doc__ = sklearn_KMeans.__doc__
        n_iter_, inertia_ = None, None
        labels_, cluster_centers_ = None, None

        if sklearn_check_version("1.2"):
            _parameter_constraints: dict = {**sklearn_KMeans._parameter_constraints}

            @_deprecate_positional_args
            def __init__(
                self,
                n_clusters=8,
                *,
                init="k-means++",
                n_init="auto" if sklearn_check_version("1.4") else "warn",
                max_iter=300,
                tol=1e-4,
                verbose=0,
                random_state=None,
                copy_x=True,
                algorithm="lloyd",
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

        elif sklearn_check_version("1.0"):

            @_deprecate_positional_args
            def __init__(
                self,
                n_clusters=8,
                *,
                init="k-means++",
                n_init=10,
                max_iter=300,
                tol=1e-4,
                verbose=0,
                random_state=None,
                copy_x=True,
                algorithm="lloyd" if sklearn_check_version("1.1") else "auto",
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
                init="k-means++",
                n_init=10,
                max_iter=300,
                tol=1e-4,
                precompute_distances="deprecated",
                verbose=0,
                random_state=None,
                copy_x=True,
                n_jobs="deprecated",
                algorithm="auto",
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
                "n_clusters": self.n_clusters,
                "init": self.init,
                "max_iter": self.max_iter,
                "tol": self.tol,
                "n_init": self.n_init,
                "verbose": self.verbose,
                "random_state": self.random_state,
            }

            self._onedal_estimator = onedal_KMeans(**onedal_params)

        def _onedal_fit_supported(self, method_name, X, y=None, sample_weight=None):
            assert method_name == "fit"

            class_name = self.__class__.__name__
            patching_status = PatchingConditionsChain(f"sklearn.cluster.{class_name}.fit")

            sample_count = _num_samples(X)
            self._algorithm = self.algorithm
            supported_algs = ["auto", "full", "lloyd", "elkan"]
            correct_count = self.n_clusters < sample_count
            if sample_weight is not None:
                if len(sample_weight) == (X.shape[0],) and (
                    np.allclose(np.asarray(sample_weight), np.ones_like(sample_weight))
                ):
                    is_sample_weight_valid = True
                else:
                    is_sample_weight_valid = False
            else:
                is_sample_weight_valid = True

            patching_status.and_conditions(
                [
                    (
                        self.algorithm in supported_algs,
                        "Only lloyd algorithm is supported, elkan is computed using lloyd",
                    ),
                    (correct_count, "n_clusters is smaller than number of samples"),
                    (
                        is_sample_weight_valid,
                        "Sample weight must be None or array of ones of length n_samples.",
                    ),
                ]
            )

            return patching_status

        def fit(self, X, y=None, sample_weight=None):
            if sklearn_check_version("1.0"):
                self._check_feature_names(X, reset=True)
            if sklearn_check_version("1.2"):
                self._validate_params()

            dispatch(
                self,
                "fit",
                {
                    "onedal": self.__class__._onedal_fit,
                    "sklearn": sklearn_KMeans.fit,
                },
                X,
                y,
                sample_weight,
            )

            return self

        def _onedal_fit(self, X, _, sample_weight, queue=None):
            assert sample_weight is None

            X = self._validate_data(
                X,
                accept_sparse="csr",
                dtype=[np.float64, np.float32],
            )

            if sklearn_check_version("1.2"):
                self._check_params_vs_input(X)
            else:
                self._check_params(X)

            self._n_features_out = self.n_clusters
            self._n_threads = _openmp_effective_n_threads()

            self._initialize_onedal_estimator()
            self._onedal_estimator.fit(X, queue=queue)

            self._save_attributes()

        def _onedal_predict_supported(self, method_name, X, sample_weight):
            assert method_name == "predict"

            class_name = self.__class__.__name__
            patching_status = PatchingConditionsChain(
                f"sklearn.cluster.{class_name}.predict"
            )

            supported_algs = ["auto", "full", "lloyd", "elkan"]

            patching_status.and_conditions(
                [
                    (
                        self.algorithm in supported_algs,
                        "Only lloyd algorithm is supported, elkan is computed using lloyd.",
                    ),
                    (
                        hasattr(self, "_onedal_estimator"),
                        "oneDAL model was not fit.",
                    ),
                ]
            )

            return patching_status

        @wrap_output_data
        def predict(
            self, X, sample_weight="deprecated" if sklearn_check_version("1.3") else None
        ):
            if sklearn_check_version("1.0"):
                self._check_feature_names(X, reset=True)
            if sklearn_check_version("1.2"):
                self._validate_params()
            return dispatch(
                self,
                "predict",
                {
                    "onedal": self.__class__._onedal_predict,
                    "sklearn": sklearn_KMeans.predict,
                },
                X,
                sample_weight,
            )

        def _onedal_predict(self, X, sample_weight=None, queue=None):
            X = self._validate_data(
                X,
                accept_sparse="csr",
                reset=False,
                dtype=[np.float64, np.float32],
            )
            if (
                sklearn_check_version("1.3")
                and isinstance(sample_weight, str)
                and sample_weight == "deprecated"
            ):
                sample_weight = None

            if sklearn_check_version("1.3") and sample_weight is not None:
                warnings.warn(
                    "'sample_weight' was deprecated in version 1.3 and "
                    "will be removed in 1.5.",
                    FutureWarning,
                )

            if not hasattr(self, "_onedal_estimator"):
                self._initialize_onedal_estimator()
                self._onedal_estimator.cluster_centers_ = self.cluster_centers_

            return self._onedal_estimator.predict(X, queue=queue)

        def _onedal_supported(self, method_name, *data):
            if method_name == "fit":
                return self._onedal_fit_supported(method_name, *data)
            if method_name == "predict":
                return self._onedal_predict_supported(method_name, *data)
            raise RuntimeError(
                f"Unknown method {method_name} in {self.__class__.__name__}"
            )

        def _onedal_gpu_supported(self, method_name, *data):
            return self._onedal_supported(method_name, *data)

        def _onedal_cpu_supported(self, method_name, *data):
            return self._onedal_supported(method_name, *data)

        @wrap_output_data
        def fit_transform(self, X, y=None, sample_weight=None):
            return self.fit(X, sample_weight=sample_weight)._transform(X)

        @wrap_output_data
        def transform(self, X):
            check_is_fitted(self)

            X = self._check_test_data(X)
            return self._transform(X)

        score = support_usm_ndarray()(sklearn_KMeans.score)

        fit.__doc__ = sklearn_KMeans.fit.__doc__
        predict.__doc__ = sklearn_KMeans.predict.__doc__
        transform.__doc__ = sklearn_KMeans.transform.__doc__
        fit_transform.__doc__ = sklearn_KMeans.fit_transform.__doc__

else:
    from daal4py.sklearn.cluster import KMeans

    logging.warning(
        "Sklearnex KMeans requires oneDAL version >= 2023.2 " "but it was not found"
    )
