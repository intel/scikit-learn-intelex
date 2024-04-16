# ===============================================================================
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
# ===============================================================================

import numpy as np
import pytest
from numpy.testing import assert_allclose

from daal4py.sklearn._utils import daal_check_version
from onedal.tests.utils._dataframes_support import (
    _as_numpy,
    _convert_to_dataframe,
    get_dataframes_and_queues,
)
from sklearnex.preview.decomposition import IncrementalPCA


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import(dataframe, queue):
    X = [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    incpca = IncrementalPCA(n_components=2)
    result = incpca.fit(X)
    assert "sklearnex" in incpca.__module__
    assert hasattr(incpca, "_onedal_estimator")
    assert_allclose(_as_numpy(result.singular_values_), [6.30061232, 0.54980396])


def check_pca_on_gold_data(incpca, dtype, is_deterministic, transformed_data):
    expected_n_components_ = 2
    expected_components_ = np.array([[0.83849224, 0.54491354], [-0.54491354, 0.83849224]])
    expected_singular_values_ = np.array([6.30061232, 0.54980396])
    expected_mean_ = np.array([0, 0])
    expected_explained_variance_ = np.array([7.93954312, 0.06045688])
    expected_explained_variance_ratio_ = np.array([0.99244289, 0.00755711])
    expected_transformed_data = np.array(
        [
            [-1.38340578, -0.2935787],
            [-2.22189802, 0.25133484],
            [-3.6053038, -0.04224385],
            [1.38340578, 0.2935787],
            [2.22189802, -0.25133484],
            [3.6053038, 0.04224385],
        ]
    )

    tol = 1e-6 if dtype == np.float32 else 1e-7

    assert incpca.n_components_ == expected_n_components_

    assert_allclose(incpca.singular_values_, expected_singular_values_, atol=tol)
    assert_allclose(incpca.mean_, expected_mean_, atol=tol)
    assert_allclose(incpca.explained_variance_, expected_explained_variance_, atol=tol)
    assert_allclose(
        incpca.explained_variance_ratio_, expected_explained_variance_ratio_, atol=tol
    )
    if is_deterministic and daal_check_version((2024, "P", 500)):
        assert_allclose(incpca.components_, expected_components_, atol=tol)
        assert_allclose(_as_numpy(transformed_data), expected_transformed_data, atol=tol)
    else:
        for i in range(incpca.n_components_):
            abs_dot_product = np.abs(
                np.dot(incpca.components_[i], expected_components_[i])
            )
            assert np.abs(abs_dot_product - 1.0) < tol

            if np.dot(incpca.components_[i], expected_components_[i]) < 0:
                assert_allclose(
                    _as_numpy(-transformed_data[i]),
                    expected_transformed_data[i],
                    atol=tol,
                )
            else:
                assert_allclose(
                    _as_numpy(transformed_data[i]), expected_transformed_data[i], atol=tol
                )


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("is_deterministic", [True, False])
@pytest.mark.parametrize("num_blocks", [1, 2, 3])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sklearnex_partial_fit_on_gold_data(
    dataframe, queue, is_deterministic, num_blocks, dtype
):

    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    X = X.astype(dtype=dtype)
    X_split = np.array_split(X, num_blocks)
    incpca = IncrementalPCA()

    for i in range(num_blocks):
        X_split_df = _convert_to_dataframe(
            X_split[i], sycl_queue=queue, target_df=dataframe
        )
        incpca.partial_fit(X_split_df)

    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    transformed_data = incpca.transform(X_df)
    check_pca_on_gold_data(incpca, dtype, is_deterministic, transformed_data)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("is_deterministic", [True, False])
@pytest.mark.parametrize("num_blocks", [1, 2, 3])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sklearnex_fit_on_gold_data(
    dataframe, queue, is_deterministic, num_blocks, dtype
):

    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    X = X.astype(dtype=dtype)
    incpca = IncrementalPCA(batch_size=X.shape[0] // num_blocks)

    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    incpca.fit(X_df)
    transformed_data = incpca.transform(X_df)

    check_pca_on_gold_data(incpca, dtype, is_deterministic, transformed_data)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("is_deterministic", [True, False])
@pytest.mark.parametrize("num_blocks", [1, 2, 3])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sklearnex_fit_transform_on_gold_data(
    dataframe, queue, is_deterministic, num_blocks, dtype
):

    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    X = X.astype(dtype=dtype)
    incpca = IncrementalPCA(batch_size=X.shape[0] // num_blocks)

    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    transformed_data = incpca.fit_transform(X_df)

    check_pca_on_gold_data(incpca, dtype, is_deterministic, transformed_data)
