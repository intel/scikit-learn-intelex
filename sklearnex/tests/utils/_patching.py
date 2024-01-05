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
from sklearn.base import ClassifierMixin, RegressorMixin, ClusterMixin, TransformerMixin
from sklearn.neighbors._base import KNeighborsMixin
from sklearnex import get_patch_map, is_patched_instance, patch_sklearn, unpatch_sklearn


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


SPECIAL_INSTANCES = [
    LocalOutlierFactor(novelty=True),
    SVC(probability=True),
    KNeighborsClassifier(algorithm="brute"),
    KNeighborsRegressor(algorithm="brute"),
    NearestNeighbors(algorithm="brute"),
]


def collect_test_info(algorithm):
    methods = []
    candidates = set(
        [i for i in dir(algorithm) if not i.startswith("_") and not i.endswith("_")]
    )
    dataset = None

    for mixin, method, data in mixin_map:
        if isinstance(algorithm, mixin):
            methods += list(candidates.intersection(set(method)))
            if data:
                dataset = data

    if not dataset:
        dataset = "classification"

    return list(set(methods)), dataset  # return only unique values


def gen_models_info():
    output = []
    for i in PATCHED_MODELS:
        estimator = PATCHED_MODELS[i]()
        methods, dataset = collect_test_info(UNPATCHED_MODELS[i]())
        output += [[estimator, j, dataset] for j in methods]

    for i in SPECIAL_INSTANCES:
        methods, dataset = collect_test_info(UNPATCHED_MODELS[j.__class__.__name__]())
        output += [[i, j, dataset] for j in methods]
    return output


TYPES = [
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

TO_SKIP = [
    # --------------- NO INFO ---------------
    r"KMeans .*transform",
    r"KMeans .*score",
    r"PCA .*score",
    r"LogisticRegression .*decision_function",
    r"LogisticRegressionCV .*decision_function",
    r"LogisticRegressionCV .*predict",
    r"LogisticRegressionCV .*predict_proba",
    r"LogisticRegressionCV .*predict_log_proba",
    r"LogisticRegressionCV .*score",
    # --------------- Scikit ---------------
    r"Ridge float16 predict",
    r"Ridge float16 score",
    r"RandomForestClassifier .*predict_proba",
    r"RandomForestClassifier .*predict_log_proba",
    r"pairwise_distances .*pairwise_distances",  # except float64
    r"roc_auc_score .*roc_auc_score",
]
