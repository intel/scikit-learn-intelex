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


import gc
import logging
import os
import tracemalloc
import types

import numpy as np
import pandas as pd
import pytest
from _utils import PATCHED_FUNCTIONS, PATCHED_MODELS, SPECIAL_INSTANCES
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, clone
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold

from onedal import _is_dpc_backend
from onedal.tests.utils._dataframes_support import (
    _convert_to_dataframe,
    get_dataframes_and_queues,
)

if _is_dpc_backend:
    from onedal import get_used_memory


class TrainTestSplitEstimator:
    def __init__(self):
        pass

    def fit(self, x, y):
        train_test_split(x, y)


class FiniteCheckEstimator:
    def __init__(self):
        pass

    def fit(self, x, y):
        _assert_all_finite(x)
        _assert_all_finite(y)


class PairwiseDistancesEstimator:
    def fit(self, x, y):
        pairwise_distances(x, metric=self.metric)


class CosineDistancesEstimator(PairwiseDistancesEstimator):
    def __init__(self):
        self.metric = "cosine"


class CorrelationDistancesEstimator(PairwiseDistancesEstimator):
    def __init__(self):
        self.metric = "correlation"


class RocAucEstimator:
    def __init__(self):
        pass

    def fit(self, x, y):
        print(roc_auc_score(y, np.zeros(shape=y.shape, dtype=np.int32)))


BANNED_ESTIMATORS = ("TSNE",)  # too slow for using in testing on common data size
ESTIMATORS = {
    k: v
    for k, v in {**PATCHED_MODELS, **SPECIAL_INSTANCES}.items()
    if not k in BANNED_ESTIMATORS
}
# NEED TO MAKE FUNCTIONS HERE USING PARTIAL


data_shapes = [(1000, 100), (2000, 50)]

EXTRA_MEMORY_THRESHOLD = 0.15
N_SPLITS = 10


def gen_clsf_data(n_samples, n_features):
    data, label = make_classification(
        n_classes=2, n_samples=n_samples, n_features=n_features, random_state=777
    )
    return (
        data,
        label,
        data.size * data.dtype.itemsize + label.size * label.dtype.itemsize,
    )


def get_traced_memory(queue=None):
    if _is_dpc_backend and queue and queue.sycl_device.is_gpu:
        return get_used_memory(queue)
    else:
        return tracemalloc.get_traced_memory()[0]


def split_train_inference(kf, x, y, estimator, queue=None):
    mem_tracks = []
    for train_index, test_index in kf.split(x):
        if isinstance(x, np.ndarray) or queue:
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
        elif isinstance(x, pd.core.frame.DataFrame):
            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        try:
            alg = estimator()
        except:
            alg = clone(estimator)

        alg.fit(x_train, y_train)
        if hasattr(alg, "predict"):
            alg.predict(x_test)
        elif hasattr(alg, "transform"):
            alg.transform(x_test)
        elif hasattr(alg, "kneighbors"):
            alg.kneighbors(x_test)
        del alg, x_train, x_test, y_train, y_test
        mem_tracks.append(get_traced_memory(queue))
    return mem_tracks


def _kfold_function_template(estimator, dataframe, data_shape, queue=None, func=None):
    tracemalloc.start()

    n_samples, n_features = data_shape
    X, y, data_memory_size = gen_clsf_data(n_samples, n_features)
    kf = KFold(n_splits=N_SPLITS)
    if func:
        X = func(X)

    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)

    mem_before = get_traced_memory(queue)
    mem_tracks = split_train_inference(kf, X, y, estimator, queue=queue)
    mem_iter_diffs = np.array(mem_tracks[1:]) - np.array(mem_tracks[:-1])
    mem_incr_mean, mem_incr_std = mem_iter_diffs.mean(), mem_iter_diffs.std()
    mem_incr_mean, mem_incr_std = round(mem_incr_mean), round(mem_incr_std)
    mem_iter_corr, _ = pearsonr(mem_tracks, list(range(len(mem_tracks))))
    if mem_iter_corr > 0.95:
        logging.warning(
            "Memory usage is steadily increasing with iterations "
            "(Pearson correlation coefficient between "
            f"memory tracks and iterations is {mem_iter_corr})\n"
            "Memory usage increase per iteration: "
            f"{mem_incr_mean}±{mem_incr_std} bytes"
        )
    mem_before_gc = get_traced_memory(queue)
    mem_diff = mem_before_gc - mem_before
    message = (
        "Size of extra allocated memory {} using garbage collector "
        f"is greater than {EXTRA_MEMORY_THRESHOLD * 100}% of input data"
        f"\n\tAlgorithm: {estimator.__name__}"
        f"\n\tInput data size: {data_memory_size} bytes"
        "\n\tExtra allocated memory size: {} bytes"
        " / {} %"
    )
    if mem_diff >= EXTRA_MEMORY_THRESHOLD * data_memory_size:
        logging.warning(
            message.format(
                "before", mem_diff, round((mem_diff) / data_memory_size * 100, 2)
            )
        )
    gc.collect()
    mem_after = get_traced_memory(queue)
    tracemalloc.stop()
    mem_diff = mem_after - mem_before

    assert mem_diff < EXTRA_MEMORY_THRESHOLD * data_memory_size, message.format(
        "after", mem_diff, round((mem_diff) / data_memory_size * 100, 2)
    )


# disable fallback check as logging impacts memory use


@pytest.mark.allow_sklearn_fallback
@pytest.mark.parametrize("order", ["F", "C"])
@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("estimator", ESTIMATORS.keys())
@pytest.mark.parametrize("data_shape", data_shapes)
def test_estimator_memory_leaks(estimator, dataframe, queue, order, data_shape):
    if order == "F":
        func = np.asfortranarray
    elif order == "C":
        func = np.ascontiguousarray
    else:
        func = None

    try:
        if _is_dpc_backend and queue and queue.sycl_device.is_gpu:
            os.environ["ZES_ENABLE_SYSMAN"] = "1"

        _kfold_function_template(
            ESTIMATORS[estimator], dataframe, data_shape, queue, func
        )

    except RuntimeError:
        pytest.skip("GPU memory tracing is not available")
    finally:
        if _is_dpc_backend and queue and queue.sycl_device.is_gpu:
            del os.environ["ZES_ENABLE_SYSMAN"]


@pytest.mark.allow_sklearn_fallback
@pytest.mark.parametrize("order", ["F", "C"])
@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("func", FUNCTIONS.keys())
@pytest.mark.parametrize("data_shape", data_shapes)
def test_function_memory_leaks(func, dataframe, queue, order, data_shape):
    if order == "F":
        func = np.asfortranarray
    elif order == "C":
        func = np.ascontiguousarray
    else:
        func = None

    try:
        if _is_dpc_backend and queue and queue.sycl_device.is_gpu:
            os.environ["ZES_ENABLE_SYSMAN"] = "1"

        _kfold_function_template(
            FUNCTIONS[funcs], dataframe, data_shape, queue, func
        )

    except RuntimeError:
        pytest.skip("GPU memory tracing is not available")
    finally:
        if _is_dpc_backend and queue and queue.sycl_device.is_gpu:
            del os.environ["ZES_ENABLE_SYSMAN"]
