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
from functools import partial
from inspect import isclass

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
    from onedal import _backend


BANNED_LIST = (
    "TSNE",  # too slow for using in testing on common data size
    "config_context",  # does not malloc
    "get_config",  # does not malloc
    "set_config",  # does not malloc
)


def gen_functions(functions):
    func_dict = functions.copy()

    roc_auc_score = func_dict.pop("roc_auc_score")
    func_dict["roc_auc_score"] = lambda y: roc_auc_score(
        y, np.zeros(shape=y.shape, dtype=np.int32)
    )

    pairwise_distances = func_dict.pop("pairwise_distances")
    func_dict["pairwise_distances(metric='cosine')"] = partial(
        pairwise_distances, metric="cosine"
    )
    func_dict["pairwise_distances(metric='correlation')"] = partial(
        pairwise_distances, metric="correlation"
    )
    return func_dict


FUNCTIONS = gen_functions(PATCHED_FUNCTIONS)

ESTIMATORS = {
    k: v
    for k, v in {**PATCHED_MODELS, **SPECIAL_INSTANCES, **FUNCTIONS}.items()
    if not k in BANNED_LIST
}

data_shapes = [
    pytest.param((1000, 100), id="(1000, 100)"),
    pytest.param((2000, 50), id="(2000, 50)"),
]

EXTRA_MEMORY_THRESHOLD = 0.15
N_SPLITS = 10
ORDER_DICT = {"F": np.asfortranarray, "C": np.ascontiguousarray}


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
        return _backend.get_used_memory(queue)
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

        if isclass(estimator) and issubclass(estimator, BaseEstimator):
            alg = estimator()
            flag = True
        elif isinstance(estimator, BaseEstimator):
            alg = clone(estimator)
            flag = True
        else:
            flag = False

        if flag:
            alg.fit(x_train, y_train)
            if hasattr(alg, "predict"):
                alg.predict(x_test)
            elif hasattr(alg, "transform"):
                alg.transform(x_test)
            elif hasattr(alg, "kneighbors"):
                alg.kneighbors(x_test)
            del alg
        else:
            for data in [x_train, y_train, x_test, y_test]:
                estimator(data)

        del x_train, x_test, y_train, y_test, flag
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
    if isinstance(estimator, BaseEstimator):
        name = str(estimator)
    else:
        name = estimator.__name__

    message = (
        "Size of extra allocated memory {} using garbage collector "
        f"is greater than {EXTRA_MEMORY_THRESHOLD * 100}% of input data"
        f"\n\tAlgorithm: {name}"
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
def test_memory_leaks(estimator, dataframe, queue, order, data_shape):
    func = ORDER_DICT[order]

    try:

        if _is_dpc_backend and queue and queue.sycl_device.is_gpu:
            status = os.getenv("ZES_ENABLE_SYSMAN")
            if status != "1":
                os.environ["ZES_ENABLE_SYSMAN"] = "1"

        _kfold_function_template(
            ESTIMATORS[estimator], dataframe, data_shape, queue, func
        )

    except RuntimeError:
        pytest.skip("GPU memory tracing is not available")

    finally:
        if _is_dpc_backend and queue and queue.sycl_device.is_gpu:
            if status is None:
                del os.environ["ZES_ENABLE_SYSMAN"]
            else:
                os.environ["ZES_ENABLE_SYSMAN"] = status
