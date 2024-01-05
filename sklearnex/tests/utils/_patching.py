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
from sklearn.neighbors._base import KNeighborsMixin
from sklearnex import get_patch_map, is_patched_instance, patch_sklearn, unpatch_sklearn
from sklearnex.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    NearestNeighbors,
    LocalOutlierFactor,
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
        al = algorithms[i]() if i in algorithms[i].__class__.__name__ else algorithms[i]

        methods = []
        candidates = set(
            [i for i in dir(al) if not i.startswith("_") and not i.endswith("_")]
        )
        dataset = None

        name = i.split("(")[0]  # handling SPECIAL_INSTANCES
        for mixin, method, data in mixin_map:
            if isinstance(UNPATCHED_MODELS[name](), mixin):
                methods += list(candidates.intersection(set(method)))
                if data:
                    dataset = data

        if not dataset:
            dataset = "classification"

        methods = list(set(methods))  # return only unique values
        output += [[i, j, dataset] for j in methods]
    return output


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
