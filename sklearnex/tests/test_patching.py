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
from contextlib import contextmanager

import numpy as np
import pytest
from _utils import (
    DTYPES,
    PATCHED_MODELS,
    SPECIAL_INSTANCES,
    UNPATCHED_MODELS,
    gen_dataset,
    gen_models_info,
)

from onedal.tests.utils._dataframes_support import get_dataframes_and_queues
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


@contextmanager
def log_sklearnex():
    try:
        log_stream = io.StringIO()
        log_handler = logging.StreamHandler(log_stream)
        sklearnex_logger = logging.getLogger("sklearnex")
        sklearnex_logger.addHandler(log_handler)
        sklearnex_logger.setLevel(logging.INFO)
        yield log_stream
    finally:
        log_handler.setLevel(logging.WARNING)
        sklearnex_logger.removeHandler(log_handler)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "dataframe, queue", get_dataframes_and_queues(dataframe_filter_="numpy")
)
@pytest.mark.parametrize("estimator, method", gen_models_info(PATCHED_MODELS))
def test_standard_estimator_patching(dataframe, queue, dtype, estimator, method):
    with log_sklearnex() as log:
        est = PATCHED_MODELS[estimator]()

        X, y = gen_dataset(est, queue=queue, target_df=dataframe, dtype=dtype)
        est.fit(X, y)

        if not hasattr(est, method):
            pytest.skip(f"sklearn available_if prevents testing {estimator}.{method}")

        if method != "score":
            getattr(est, method)(X)
        else:
            est.score(X, y)

        result = log.getvalue().strip().split("\n")

    assert all(
        [
            "running accelerated version" in i or "fallback to original Scikit-learn" in i
            for i in result
        ]
    ), f"sklearnex patching issue in {estimator}.{method} with log: \n" + "\n".join(
        result
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "dataframe, queue", get_dataframes_and_queues(dataframe_filter_="numpy")
)
@pytest.mark.parametrize("estimator, method", gen_models_info(SPECIAL_INSTANCES))
def test_special_estimator_patching(dataframe, queue, dtype, estimator, method):
    # prepare logging
    with log_sklearnex() as log:
        est = SPECIAL_INSTANCES[estimator]

        X, y = gen_dataset(est, queue=queue, target_df=dataframe, dtype=dtype)
        est.fit(X, y)

        if not hasattr(est, method):
            pytest.skip(f"sklearn available_if prevents testing {estimator}.{method}")

        if method != "score":
            getattr(est, method)(X)
        else:
            est.score(X, y)

        result = log.getvalue().strip().split("\n")

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
