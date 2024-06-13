# ==============================================================================
# Copyright 2024 Intel Corporation
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

from functools import partial
from inspect import getattr_static, isclass

import numpy as np
from scipy import sparse as sp
from sklearn import clone
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
from sklearnex import get_patch_map, patch_sklearn, sklearn_is_patched, unpatch_sklearn
from sklearnex.linear_model import LogisticRegression
from sklearnex.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    LocalOutlierFactor,
    NearestNeighbors,
)
from sklearnex.svm import SVC, NuSVC


def _load_all_models(with_sklearnex=True, estimator=True):
    # insure that patch state is correct as dictated by patch_sklearn boolean
    # and return it to the previous state no matter what occurs.
    already_patched_map = sklearn_is_patched(return_map=True)
    already_patched = any(already_patched_map.values())
    try:
        if with_sklearnex:
            patch_sklearn()
        elif already_patched:
            unpatch_sklearn()

        models = {}
        for patch_infos in get_patch_map().values():
            candidate = getattr(patch_infos[0][0][0], patch_infos[0][0][1], None)
            if candidate is not None and isclass(candidate) == estimator:
                if not estimator or issubclass(candidate, BaseEstimator):
                    models[patch_infos[0][0][1]] = candidate
    finally:
        if with_sklearnex:
            unpatch_sklearn()
        # both branches are now in an unpatched state, repatch as necessary
        if already_patched:
            patch_sklearn(name=[i for i in already_patched_map if already_patched_map[i]])

    return models


PATCHED_MODELS = _load_all_models(with_sklearnex=True)
UNPATCHED_MODELS = _load_all_models(with_sklearnex=False)

PATCHED_FUNCTIONS = _load_all_models(with_sklearnex=True, estimator=False)
UNPATCHED_FUNCTIONS = _load_all_models(with_sklearnex=False, estimator=False)

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


class _sklearn_clone_dict(dict):

    def __getitem__(self, key):
        return clone(super().__getitem__(key))


SPECIAL_INSTANCES = _sklearn_clone_dict(
    {
        str(i): i
        for i in [
            LocalOutlierFactor(novelty=True),
            SVC(probability=True),
            NuSVC(probability=True),
            KNeighborsClassifier(algorithm="brute"),
            KNeighborsRegressor(algorithm="brute"),
            NearestNeighbors(algorithm="brute"),
            LogisticRegression(solver="newton-cg"),
        ]
    }
)


def gen_models_info(algorithms):
    output = []
    for i in algorithms:

        if i in PATCHED_MODELS:
            est = PATCHED_MODELS[i]
        elif isinstance(algorithms[i], BaseEstimator):
            est = algorithms[i].__class__
        else:
            raise KeyError(f"Unrecognized sklearnex estimator: {i}")

        methods = set(dir(est)) - set(dir(BaseEstimator))
        methods = set(
            [i for i in methods if not i.startswith("_") and not i.endswith("_")]
        )
        methods = set(
            [i for i in methods if callable(getattr_static(est, i))]
        )
        methods = methods - {"fit"}

        output += [[i, j] for j in methods] if methods else [[i, None]]

    # In the case that no methods are available, set method to None.
    # This will allow estimators without mixins to still test the fit
    # method in various tests.
    return output


def gen_dataset_type(est):
    # est should be an estimator or estimator class
    # dataset initialized to classification, but will be swapped
    # for other types as necessary
    dataset = "classification"
    estimator = est.__class__ if isinstance(est, BaseEstimator) else est

    for mixin, _, data in mixin_map:
        if issubclass(estimator, mixin) and data is not None:
            dataset = data
    return dataset


_dataset_dict = {
    "classification": [partial(load_iris, return_X_y=True)],
    "regression": [partial(load_diabetes, return_X_y=True)],
}


def gen_dataset(
    est,
    datasets=_dataset_dict,
    sparse=False,
    queue=None,
    target_df=None,
    dtype=None,
):
    dataset_type = gen_dataset_type(est)
    output = []
    # load data
    flag = dtype is None

    for func in datasets[dataset_type]:
        X, y = func()
        if flag:
            dtype = X.dtype if hasattr(X, "dtype") else np.float64

        if sparse:
            X = sp.csr_matrix(X)
        else:
            X = _convert_to_dataframe(
                X, sycl_queue=queue, target_df=target_df, dtype=dtype
            )
            y = _convert_to_dataframe(
                y, sycl_queue=queue, target_df=target_df, dtype=dtype
            )
        output += [[X, y]]
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
