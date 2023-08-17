#!/usr/bin/env python
# ===============================================================================
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
# ===============================================================================

# from daal4py.sklearn.cluster import DBSCAN

import numbers
from abc import ABC

import numpy as np
from scipy import sparse as sp
from sklearn.cluster import DBSCAN as sklearn_DBSCAN
from sklearn.utils import check_array
from sklearn.utils.validation import _check_sample_weight

# TODO:
# move the checks to onedal4py side.
import daal4py
from daal4py.sklearn._utils import (
    PatchingConditionsChain,
    getFPType,
    make2d,
    sklearn_check_version,
)
from onedal.cluster import DBSCAN as onedal_DBSCAN

from .._device_offload import dispatch, wrap_output_data

if sklearn_check_version("1.1") and not sklearn_check_version("1.2"):
    from sklearn.utils import check_scalar


class BaseDBSCAN(ABC):
    def _save_attributes(self):
        assert hasattr(self, "_onedal_estimator")

        self.labels_ = self._onedal_estimator.labels_
        self.core_sample_indices_ = self._onedal_estimator.core_sample_indices_
        self.components_ = self._onedal_estimator.components_
        self.n_features_in_ = self._onedal_estimator.n_features_in_


class DBSCAN(sklearn_DBSCAN, BaseDBSCAN):
    __doc__ = sklearn_DBSCAN.__doc__

    # TODO:
    # move scalar checks here.
    # if sklearn_check_version("1.1"):
    #    _parameter_constraints: dict = {
    #        **sklearn_DBSCAN._parameter_constraints,
    #        "eps": [Interval(numbers.Integral, 2, None, closed="left")],
    #        "min_bin_size": [Interval(numbers.Integral, 1, None, closed="left")],
    #        "splitter_mode": [StrOptions({"best", "random"})],
    #    }

    def __init__(
        self,
        eps=0.5,
        *,
        min_samples=5,
        metric="euclidean",
        metric_params=None,
        algorithm="auto",
        leaf_size=30,
        p=None,
        n_jobs=None,
    ):
        super(DBSCAN, self).__init__(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            metric_params=metric_params,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            n_jobs=n_jobs,
        )
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.metric_params = metric_params
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.n_jobs = n_jobs

    def _onedal_fit(self, X, y, sample_weight=None, queue=None):
        # TODO:
        # use `_parameter_constraints` instead.
        if sklearn_check_version("1.2"):
            self._validate_params()
        elif sklearn_check_version("1.1"):
            check_scalar(
                self.eps,
                "eps",
                target_type=numbers.Real,
                min_val=0.0,
                include_boundaries="neither",
            )
            check_scalar(
                self.min_samples,
                "min_samples",
                target_type=numbers.Integral,
                min_val=1,
                include_boundaries="left",
            )
            check_scalar(
                self.leaf_size,
                "leaf_size",
                target_type=numbers.Integral,
                min_val=1,
                include_boundaries="left",
            )
            if self.p is not None:
                check_scalar(
                    self.p,
                    "p",
                    target_type=numbers.Real,
                    min_val=0.0,
                    include_boundaries="left",
                )
            if self.n_jobs is not None:
                check_scalar(self.n_jobs, "n_jobs", target_type=numbers.Integral)
        else:
            if self.eps <= 0.0:
                raise ValueError(f"eps == {self.eps}, must be > 0.0.")

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        onedal_params = {
            "eps": self.eps,
            "min_samples": self.min_samples,
            "metric": self.metric,
            "metric_params": self.metric_params,
            "algorithm": self.algorithm,
            "leaf_size": self.leaf_size,
            "p": self.p,
            "n_jobs": self.n_jobs,
        }
        self._onedal_estimator = onedal_DBSCAN(**onedal_params)

        self._onedal_estimator.fit(X, y=y, sample_weight=sample_weight, queue=queue)
        self._save_attributes()

    def _onedal_cpu_supported(self, method_name, *data):
        if method_name == "fit":
            X, y, sample_weight = data
            if self.algorithm not in ["auto", "brute"]:
                return False
            elif not (
                self.metric == "euclidean" or (self.metric == "minkowski" and self.p == 2)
            ):
                return False
            elif sp.issparse(X):
                return False
            else:
                return True
        raise RuntimeError(f"Unknown method {method_name} in {self.__class__.__name__}")

    def _onedal_gpu_supported(self, method_name, *data):
        if method_name == "fit":
            X, y, sample_weight = data
            if self.algorithm not in ["auto", "brute"]:
                return False
            elif not (
                self.metric == "euclidean" or (self.metric == "minkowski" and self.p == 2)
            ):
                return False
            elif sp.issparse(X):
                return False
            else:
                return True
        raise RuntimeError(f"Unknown method {method_name} in {self.__class__.__name__}")

    # @wrap_output_data
    def fit(self, X, y=None, sample_weight=None):
        dispatch(
            self,
            "fit",
            {
                "onedal": self.__class__._onedal_fit,
                "sklearn": super(DBSCAN, self).fit,
            },
            X,
            y,
            sample_weight,
        )

        return self
