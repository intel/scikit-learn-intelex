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

import os
import subprocess
import sys
from contextlib import contextmanager
from inspect import isclass

import numpy as np
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    ClusterMixin,
    OutlierMixin,
    RegressorMixin,
    TransformerMixin,
)
from sklearn.datasets import load_diabetes, load_iris
from sklearn.neighbors._base import KNeighborsMixin

from onedal.tests.utils._dataframes_support import _convert_to_dataframe
from sklearnex import get_patch_map, is_patched_instance, patch_sklearn, unpatch_sklearn
from sklearnex.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    LocalOutlierFactor,
    NearestNeighbors,
)
from sklearnex.svm import SVC


def _load_all_models(patched):
    if patched:
        patch_sklearn()

    models = {}
    for patch_infos in get_patch_map().values():
        maybe_class = getattr(patch_infos[0][0][0], patch_infos[0][0][1])
        if (
            maybe_class is not None
            and isclass(maybe_class)
            and issubclass(maybe_class, BaseEstimator)
        ):
            models[patch_infos[0][0][1]] = maybe_class

    if patched:
        unpatch_sklearn()

    return models


PATCHED_MODELS = _load_all_models(patched=True)
UNPATCHED_MODELS = _load_all_models(patched=False)

mixin_map = [
    [
        ClassifierMixin,
        ["decision_function", "predict", "predict_proba", "predict_log_proba", "score"],
        "classification",
    ],
    [RegressorMixin, ["predict", "score"], "regression"],
    [ClusterMixin, ["fit_predict"], "classification"],
    [TransformerMixin, ["fit_transform", "transform", "score"], "classification"],
    [OutlierMixin, ["fit_predict", "predict"], "classification"],
    [KNeighborsMixin, ["kneighbors"], None],
]


SPECIAL_INSTANCES = {
    i.__str__(): i
    for i in [
        LocalOutlierFactor(novelty=True),
        SVC(probability=True),
        KNeighborsClassifier(algorithm="brute"),
        KNeighborsRegressor(algorithm="brute"),
        NearestNeighbors(algorithm="brute"),
    ]
}


def gen_models_info(algorithms):
    output = []
    for i in algorithms:
        # split handles SPECIAL_INSTANCES or custom inputs
        # custom sklearn inputs must be a dict of estimators
        # with keys set by the __str__ method
        est = UNPATCHED_MODELS[i.split("(")[0]]()

        methods = []
        candidates = set(
            [i for i in dir(est) if not i.startswith("_") and not i.endswith("_")]
        )

        for mixin, method, _ in mixin_map:
            if isinstance(est, mixin):
                methods += list(candidates.intersection(set(method)))

        methods = list(set(methods))  # return only unique values
        output += [[i, j] for j in methods]
    return output


def gen_dataset(estimator, queue=None, target_df=None, dtype=np.float64):
    dataset = None
    name = estimator.__class__.__name__
    est = UNPATCHED_MODELS[name]()
    for mixin, _, data in mixin_map:
        if isinstance(est, mixin) and data is not None:
            dataset = data
    # load data
    if dataset == "classification" or dataset is None:
        X, y = load_iris(return_X_y=True)
    elif dataset == "regression":
        X, y = load_diabetes(return_X_y=True)
    else:
        raise ValueError("Unknown dataset type")

    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=target_df, dtype=dtype)
    y = _convert_to_dataframe(y, sycl_queue=queue, target_df=target_df, dtype=dtype)
    return X, y


DTYPES = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.float16,
    np.float32,
    np.float64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]
