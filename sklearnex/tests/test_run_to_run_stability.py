# ===============================================================================
# Copyright 2020 Intel Corporation
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

import random
from functools import partial
from numbers import Number

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_iris,
    make_classification,
    make_regression,
)

import daal4py as d4p
from onedal.tests.utils._dataframes_support import _as_numpy
from sklearnex.cluster import DBSCAN, KMeans
from sklearnex.decomposition import PCA
from sklearnex.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearnex.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearnex.manifold import TSNE
from sklearnex.metrics import pairwise_distances, roc_auc_score
from sklearnex.model_selection import train_test_split
from sklearnex.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    LocalOutlierFactor,
    NearestNeighbors,
)
from sklearnex.svm import SVC, SVR, NuSVC, NuSVR

from ._utils import (
    PATCHED_MODELS,
    SPECIAL_INSTANCES,
    _sklearn_clone_dict,
    gen_dataset,
    gen_models_info,
)

# to reproduce errors even in CI
d4p.daalinit(nthreads=100)


def get_class_name(x):
    return x.__class__.__name__


def method_processing(X, clf, methods):
    res = []
    name = []
    for i in methods:
        if i == "predict":
            res.append(clf.predict(X))
            name.append(get_class_name(clf) + ".predict(X)")
        elif i == "predict_proba":
            res.append(clf.predict_proba(X))
            name.append(get_class_name(clf) + ".predict_proba(X)")
        elif i == "decision_function":
            res.append(clf.decision_function(X))
            name.append(get_class_name(clf) + ".decision_function(X)")
        elif i == "kneighbors":
            dist, idx = clf.kneighbors(X)
            res.append(dist)
            name.append("dist")
            res.append(idx)
            name.append("idx")
        elif i == "fit_predict":
            predict = clf.fit_predict(X)
            res.append(predict)
            name.append(get_class_name(clf) + ".fit_predict")
        elif i == "fit_transform":
            res.append(clf.fit_transform(X))
            name.append(get_class_name(clf) + ".fit_transform")
        elif i == "transform":
            res.append(clf.transform(X))
            name.append(get_class_name(clf) + ".transform(X)")
        elif i == "get_covariance":
            res.append(clf.get_covariance())
            name.append(get_class_name(clf) + ".get_covariance()")
        elif i == "get_precision":
            res.append(clf.get_precision())
            name.append(get_class_name(clf) + ".get_precision()")
        elif i == "score_samples":
            res.append(clf.score_samples(X))
            name.append(get_class_name(clf) + ".score_samples(X)")
    return res, name


def func(X, Y, clf, methods):
    clf.fit(X, Y)
    res, name = method_processing(X, clf, methods)

    for i in clf.__dict__.keys():
        ans = getattr(clf, i)
        if isinstance(ans, (bool, float, int, np.ndarray, np.float64)):
            if isinstance(ans, np.ndarray) and None in ans:
                continue
            res.append(ans)
            name.append(get_class_name(clf) + "." + i)
    return res, name


_dataset_dict = {
    "classification": [
        partial(load_iris, return_X_y=True),
        partial(load_breast_cancer, return_X_y=True),
    ],
    "regression": [
        partial(load_diabetes, return_X_y=True),
        partial(
            make_regression, n_samples=500, n_features=10, noise=64.0, random_state=42
        ),
    ],
}


def eval_method(X, y, estimator, method):
    estimator.fit(X, y)

    if method:
        if method != "score":
            res = getattr(est, method)(X)
        else:
            res = est.score(X, y)

        # if estimator follows sklearn design rules, then set attributes should have a
        # trailing underscore
        attributes = [i for i in dir(est) if not i.startswith("_") and i.endswith("_")]
        results = [getattr(est, i) for i in attributes] + [_as_numpy(i) for i in res]
        attributes += [method for i in res]
    return results, attributes


def _run_test(estimator, method, datasets):

    for X, y in datasets:
        baseline, attributes = eval_method(X, y, estimator, method)

        for i in range(10):
            res, _ = eval_method(X, y, estimator, methods)

            for r, b, n in zip(res, baseline, attributes):
                if (
                    isinstance(b, Number)
                    or hasattr(b, "__array__")
                    or hasattr(b, "__array_namespace__")
                    or hasattr(b, "__sycl_usm_ndarray__")
                ):
                    assert_allclose(
                        r, b, rtol=0.0, atol=0.0, err_msg=str(n + " is incorrect")
                    )


SPARSE_INSTANCES = _sklearn_clone_dict(
    {
        str(i): i
        for i in [
            SVC(),
            KMeans(),
            KMeans(init="random"),
        ]
    }
)

MODELS_INFO = [
    {
        "model": KNeighborsClassifier(
            n_neighbors=10, algorithm="brute", weights="uniform"
        ),
        "methods": ["predict", "predict_proba", "kneighbors"],
        "dataset": "classifier",
    },
    {
        "model": KNeighborsClassifier(
            n_neighbors=10, algorithm="brute", weights="distance"
        ),
        "methods": ["predict", "predict_proba", "kneighbors"],
        "dataset": "classifier",
    },
    {
        "model": KNeighborsClassifier(
            n_neighbors=10, algorithm="kd_tree", weights="uniform"
        ),
        "methods": ["predict", "predict_proba", "kneighbors"],
        "dataset": "classifier",
    },
    {
        "model": KNeighborsClassifier(
            n_neighbors=10, algorithm="kd_tree", weights="distance"
        ),
        "methods": ["predict", "predict_proba", "kneighbors"],
        "dataset": "classifier",
    },
    {
        "model": KNeighborsRegressor(
            n_neighbors=10, algorithm="kd_tree", weights="distance"
        ),
        "methods": ["predict", "kneighbors"],
        "dataset": "regression",
    },
    {
        "model": KNeighborsRegressor(
            n_neighbors=10, algorithm="kd_tree", weights="uniform"
        ),
        "methods": ["predict", "kneighbors"],
        "dataset": "regression",
    },
    {
        "model": KNeighborsRegressor(
            n_neighbors=10, algorithm="brute", weights="distance"
        ),
        "methods": ["predict", "kneighbors"],
        "dataset": "regression",
    },
    {
        "model": KNeighborsRegressor(
            n_neighbors=10, algorithm="brute", weights="uniform"
        ),
        "methods": ["predict", "kneighbors"],
        "dataset": "regression",
    },
    {
        "model": NearestNeighbors(n_neighbors=10, algorithm="brute"),
        "methods": ["kneighbors"],
        "dataset": "blobs",
    },
    {
        "model": NearestNeighbors(n_neighbors=10, algorithm="kd_tree"),
        "methods": ["kneighbors"],
        "dataset": "blobs",
    },
    {
        "model": LocalOutlierFactor(n_neighbors=10, novelty=False),
        "methods": ["fit_predict"],
        "dataset": "blobs",
    },
    {
        "model": LocalOutlierFactor(n_neighbors=10, novelty=True),
        "methods": ["predict"],
        "dataset": "blobs",
    },
    {
        "model": DBSCAN(algorithm="brute", n_jobs=-1),
        "methods": [],
        "dataset": "blobs",
    },
    {
        "model": SVC(kernel="rbf"),
        "methods": ["predict", "decision_function"],
        "dataset": "classifier",
    },
    {
        "model": SVC(kernel="rbf"),
        "methods": ["predict", "decision_function"],
        "dataset": "sparse",
    },
    {
        "model": NuSVC(kernel="rbf"),
        "methods": ["predict", "decision_function"],
        "dataset": "classifier",
    },
    {
        "model": SVR(kernel="rbf"),
        "methods": ["predict"],
        "dataset": "regression",
    },
    {
        "model": NuSVR(kernel="rbf"),
        "methods": ["predict"],
        "dataset": "regression",
    },
    {
        "model": TSNE(random_state=0),
        "methods": ["fit_transform"],
        "dataset": "classifier",
    },
    {
        "model": KMeans(random_state=0, init="k-means++"),
        "methods": ["predict"],
        "dataset": "blobs",
    },
    {
        "model": KMeans(random_state=0, init="random"),
        "methods": ["predict"],
        "dataset": "blobs",
    },
    {
        "model": KMeans(random_state=0, init="k-means++"),
        "methods": ["predict"],
        "dataset": "sparse",
    },
    {
        "model": KMeans(random_state=0, init="random"),
        "methods": ["predict"],
        "dataset": "sparse",
    },
    {
        "model": ElasticNet(random_state=0),
        "methods": ["predict"],
        "dataset": "regression",
    },
    {
        "model": Lasso(random_state=0),
        "methods": ["predict"],
        "dataset": "regression",
    },
    {
        "model": PCA(n_components=0.5, svd_solver="covariance_eigh", random_state=0),
        "methods": ["transform", "get_covariance", "get_precision", "score_samples"],
        "dataset": "classifier",
    },
    {
        "model": RandomForestClassifier(
            random_state=0, oob_score=True, max_samples=0.5, max_features="sqrt"
        ),
        "methods": ["predict", "predict_proba"],
        "dataset": "classifier",
    },
    {
        "model": LogisticRegression(random_state=0, solver="newton-cg", max_iter=1000),
        "methods": ["predict", "predict_proba"],
        "dataset": "classifier",
    },
    {
        "model": LogisticRegression(random_state=0, solver="lbfgs", max_iter=1000),
        "methods": ["predict", "predict_proba"],
        "dataset": "classifier",
    },
    {
        "model": LogisticRegressionCV(
            random_state=0, solver="newton-cg", n_jobs=-1, max_iter=1000
        ),
        "methods": ["predict", "predict_proba"],
        "dataset": "classifier",
    },
    {
        "model": RandomForestRegressor(
            random_state=0, oob_score=True, max_samples=0.5, max_features="sqrt"
        ),
        "methods": ["predict"],
        "dataset": "regression",
    },
    {
        "model": LinearRegression(),
        "methods": ["predict"],
        "dataset": "regression",
    },
    {
        "model": Ridge(random_state=0),
        "methods": ["predict"],
        "dataset": "regression",
    },
]

TO_SKIP = [
    "TSNE",  # Absolute diff is 1e-10, potential problem in KNN,
    # will be fixed for next release. (UPD. KNN is fixed but there is a problem
    # with stability of stock sklearn. It is already stable in master, so, we
    # need to wait for the next sklearn release)
    "LogisticRegression",  # Absolute diff is 1e-8, will be fixed for next release
    "LogisticRegressionCV",  # Absolute diff is 1e-10, will be fixed for next release
    "RandomForestRegressor",  # Absolute diff is 1e-14 in OOB score,
    # will be fixed for next release
]


@pytest.mark.parametrize("dataframe, queue", get_dataframes_and_queues("numpy"))
@pytest.mark.parametrize("estimator, method", gen_models_info(PATCHED_MODELS))
def test_standard_estimator_stability(estimator, method, dataframe, queue):
    if estimator in TO_SKIP:
        pytest.skip(f"stability not guaranteed for {estimator}")

    est = PATCHED_MODELS[estimator]()
    params = est.get_params().copy()
    if "random_state" in params:
        params["random_state"] = 0
        est.set_params(params)

    datasets = gen_dataset(
        est, datasets=_dataset_dict, queue=queue, target_df=dataframe, dtype=dtype
    )
    _run_test(est, method, datasets)


@pytest.mark.parametrize("dataframe, queue", get_dataframes_and_queues("numpy"))
@pytest.mark.parametrize("estimator, method", gen_models_info(SPECIAL_INSTANCES))
def test_special_estimator_stability(estimator, method, dataframe, queue):
    if estimator in TO_SKIP:
        pytest.skip(f"stability not guaranteed for {estimator}")

    est = SPECIAL_INSTANCES[estimator]
    params = est.get_params().copy()
    if "random_state" in params:
        params["random_state"] = 0
        est.set_params(params)

    datasets = gen_dataset(
        est, datasets=_dataset_dict, queue=queue, target_df=dataframe, dtype=dtype
    )
    _run_test(est, method, datasets)


@pytest.mark.parametrize("dataframe, queue", get_dataframes_and_queues("numpy"))
@pytest.mark.parametrize("estimator, method", gen_models_info(SPARSE_INSTANCES))
def test_sparse_estimator_stability(estimator, method, dataframe, queue):
    if estimator in TO_SKIP:
        pytest.skip(f"stability not guaranteed for {estimator}")

    est = SPARSE_INSTANCES[estimator]
    params = est.get_params().copy()
    if "random_state" in params:
        params["random_state"] = 0
        est.set_params(params)

    datasets = gen_dataset(
        est, datasets=_dataset_dict, queue=queue, target_df=dataframe, dtype=dtype
    )
    _run_test(est, method, datasets)


@pytest.mark.parametrize("features", range(5, 10))
def test_train_test_split(features):
    X, y = make_classification(
        n_samples=4000,
        n_features=features,
        n_informative=features,
        n_redundant=0,
        n_clusters_per_class=8,
        random_state=0,
    )
    (
        baseline_X_train,
        baseline_X_test,
        baseline_y_train,
        baseline_y_test,
    ) = train_test_split(X, y, test_size=0.33, random_state=0)
    baseline = [baseline_X_train, baseline_X_test, baseline_y_train, baseline_y_test]
    for _ in range(10):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=0
        )
        res = [X_train, X_test, y_train, y_test]
        for a, b in zip(res, baseline):
            np.testing.assert_allclose(
                a, b, rtol=0.0, atol=0.0, err_msg=str("train_test_split is incorrect")
            )


@pytest.mark.parametrize("metric", ["cosine", "correlation"])
def test_pairwise_distances(metric):
    X = np.random.rand(1000)
    X = np.array(X, dtype=np.float64)
    baseline = pairwise_distances(X.reshape(1, -1), metric=metric)
    for _ in range(5):
        res = pairwise_distances(X.reshape(1, -1), metric=metric)
        for a, b in zip(res, baseline):
            np.testing.assert_allclose(
                a, b, rtol=0.0, atol=0.0, err_msg=str("pairwise_distances is incorrect")
            )


@pytest.mark.parametrize("array_size", [100, 1000, 10000])
def test_roc_auc(array_size):
    a = [random.randint(0, 1) for i in range(array_size)]
    b = [random.randint(0, 1) for i in range(array_size)]
    baseline = roc_auc_score(a, b)
    for _ in range(5):
        res = roc_auc_score(a, b)
        np.testing.assert_allclose(
            baseline, res, rtol=0.0, atol=0.0, err_msg=str("roc_auc is incorrect")
        )
