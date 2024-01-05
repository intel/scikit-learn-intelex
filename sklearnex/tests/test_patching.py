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

import re

import pytest
from .utils._patching import DTYPES, PATCHED_MODELS, UNPATCHED_MODELS, TO_SKIP, gen_models_info
from sklearn.base import BaseEstimator
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


@pytest.mark.parametrize("estimator, method, dataset", gen_models_info())
@pytest.mark.parametrize("dtype", DTYPES)
def test_estimator_patching(estimator, method, dataset, dtype):
    #load data
    if dataset == "classification":
        X, y = load_iris(return_X_y=True)
    elif dataset == "regression":
        X, y = load_diabetes(return_X_y=True)
    else:
        raise ValueError("Unknown dataset type")

    X = X.astype(dtype)
    y = y.astype(dtype)

    #prepare logging
    log_stream = io.StringIO()
    log_handler = logging.StreamHandler(log_stream)
    sklearnex_logger = logging.getLogger("sklearnex")
    sklearnex_logger.addHandler(log_handler)
    sklearnex_logger.setLevel(logging.INFO)

    estimator.fit(X, y)
    if method != "score":
        getattr(estimator, method)(X)
    else:
        estimator.score(X, y)

    result = log_stream.getvalue().decode("utf-8").split("\n")
    sklearnex_logger.getLogger("sklearnex").setLevel(logging.WARNING)

    print(result)
    assert False


@pytest.mark.parametrize("name", PATCHED_MODELS.keys())
def test_is_patched_instance(name):
    patched = PATCHED_MODELS[name]()
    unpatched = UNPATCHED_MODELS[name]()
    assert is_patched_instance(patched), f"{patched} is a patched instance"
    assert not is_patched_instance(unpatched), f"{unpatched} is an unpatched instance"
