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

import io
import logging
import re

import pytest
from utils._patching import (
    DTYPES,
    PATCHED_MODELS,
    UNPATCHED_MODELS,
    gen_models_info,
)
from sklearn.datasets import load_diabetes, load_iris, make_regression

from sklearnex import get_patch_map, is_patched_instance, patch_sklearn, unpatch_sklearn
from sklearnex.metrics import pairwise_distances, roc_auc_score


def run_utils():
    # pairwise_distances
    for metric in ["cosine", "correlation"]:
        for t in TYPES:
            X = np.random.rand(1000)
            X = np.array(X, dtype=t)
            print("pairwise_distances", t.__name__)
            _ = pairwise_distances(X.reshape(1, -1), metric=metric)
            logging.info("pairwise_distances")
    # roc_auc_score
    for t in [np.float32, np.float64]:
        a = [random.randint(0, 1) for i in range(1000)]
        b = [random.randint(0, 1) for i in range(1000)]
        a = np.array(a, dtype=t)
        b = np.array(b, dtype=t)
        print("roc_auc_score", t.__name__)
        _ = roc_auc_score(a, b)
        logging.info("roc_auc_score")


def _load_dataset(dataset, dtype):
    # load data
    if dataset == "classification":
        X, y = load_iris(return_X_y=True)
    elif dataset == "regression":
        X, y = load_diabetes(return_X_y=True)
    else:
        raise ValueError("Unknown dataset type")

    return X.astype(dtype), y.astype(dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("estimator, method, dataset", gen_models_info(PATCHED_MODELS))
def test_standard_estimator_patching(dtype, estimator, method, dataset):
    X, y = _load_dataset(dataset, dtype)
    # prepare logging
    log_stream = io.StringIO()
    log_handler = logging.StreamHandler(log_stream)
    sklearnex_logger = logging.getLogger("sklearnex")
    sklearnex_logger.addHandler(log_handler)
    sklearnex_logger.setLevel(logging.INFO)

    est = PATCHED_MODELS[estimator]().fit(X, y)
    if not hasattr(est, method):
        pytest.skip(f"sklearn available_if prevents testing {estimator}.{method}")

    if method != "score":
        getattr(est, method)(X)
    else:
        est.score(X, y)

    result = log_stream.getvalue().strip().split("\n")
    sklearnex_logger.setLevel(logging.WARNING)

    assert all(
        [
            "running accelerated version" in i or "fallback to original Scikit-learn" in i
            for i in result
        ]
    ), f"sklearnex patching issue in {estimator}.{method} with log: \n" + "\n".join(
        result
    )


@pytest.mark.parametrize("name", PATCHED_MODELS.keys())
def test_is_patched_instance(name):
    patched = PATCHED_MODELS[name]()
    unpatched = UNPATCHED_MODELS[name]()
    assert is_patched_instance(patched), f"{patched} is a patched instance"
    assert not is_patched_instance(unpatched), f"{unpatched} is an unpatched instance"
