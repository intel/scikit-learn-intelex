# ===============================================================================
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
# ===============================================================================

import numpy as np
from numpy.testing import assert_allclose
import pytest
from onedal.tests.utils._dataframes_support import (
    _as_numpy,
    _convert_to_dataframe,
    get_dataframes_and_queues,
)

def test_sklearnex_import():
    from sklearnex.manifold import TSNE

    X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    tsne = TSNE(n_components=2, perplexity=2.0).fit(X)
    assert "daal4py" in tsne.__module__

from sklearnex.manifold import TSNE

@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_tsne_import(dataframe, queue):
    """Test TSNE compatibility with different backends and queues, and validate sklearnex module."""
    X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    tsne = TSNE(n_components=2, perplexity=2.0).fit(X_df)
    assert "daal4py" in tsne.__module__
    assert hasattr(tsne, "n_components"), "TSNE missing 'n_components' attribute."
    assert tsne.n_components == 2, "TSNE 'n_components' attribute is incorrect."

def test_valid_tsne_functionality():
    """Test TSNE with valid data: basic functionality, random data, reproducibility, and edge cases."""
    # Test basic functionality
    X_basic = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    tsne = TSNE(n_components=2, perplexity=2.0).fit(X_basic)
    assert tsne.embedding_.shape == (4, 2)
    
    # Test with random data
    np.random.seed(42)
    X_random = np.random.rand(100, 10)
    tsne_random = TSNE(n_components=2, perplexity=30.0).fit(X_random)
    assert tsne_random.embedding_.shape == (100, 2)

    # Test reproducibility
    X_repro = np.random.rand(50, 10)
    tsne_1 = TSNE(n_components=2, random_state=42).fit_transform(X_repro)
    tsne_2 = TSNE(n_components=2, random_state=42).fit_transform(X_repro)
    assert_allclose(tsne_1, tsne_2, rtol=1e-5)

    # Test perplexity edge case (close to dataset size)
    X_perplexity = np.random.rand(10, 5)
    tsne_perplexity = TSNE(n_components=2, perplexity=9).fit(X_perplexity)
    assert tsne_perplexity.embedding_.shape == (10, 2)

    # Test large data
    X_large = np.random.rand(1000, 50)
    tsne_large = TSNE(n_components=2, perplexity=50.0).fit(X_large)
    assert tsne_large.embedding_.shape == (1000, 2)

    # Test valid minimal data
    X_valid = np.array([[0, 0], [1, 1], [2, 2]])
    tsne_valid = TSNE(n_components=2, perplexity=2).fit(X_valid)
    assert tsne_valid.embedding_.shape == (3, 2)

def test_tsne_edge_cases_and_errors():
    """Test TSNE with invalid, constant, and edge-case data."""
    # Edge case: constant data
    X_constant = np.ones((10, 10))
    with pytest.raises(ValueError) as excinfo:
        TSNE(n_components=2, perplexity=20).fit(X_constant)
    assert "perplexity must be less than n_samples" in str(excinfo.value)

    # Edge case: empty data
    X_empty = np.empty((0, 10))
    with pytest.raises(ValueError):
        TSNE(n_components=2).fit(X_empty)

    # Edge case: data with NaN or infinite values
    X_invalid = np.array([[0, 0], [1, np.nan], [2, np.inf]])
    with pytest.raises(ValueError):
        TSNE(n_components=2).fit(X_invalid)

@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("num_blocks", [1, 2, 4])
def test_tsne_full_fit_with_blocks(dataframe, queue, dtype, num_blocks):
    """Test TSNE fitted on the full dataset, after splitting into blocks."""
    np.random.seed(42)
    X = np.random.rand(100, 20).astype(dtype)  # 100 samples, 20 features
    X_blocks = np.array_split(X, num_blocks)   # Split into `num_blocks`

    # Combine blocks back into a single dataset
    X_combined = np.vstack(X_blocks)
    X_df = _convert_to_dataframe(X_combined, sycl_queue=queue, target_df=dataframe)

    # Fit TSNE on the combined dataset
    tsne = TSNE(n_components=2, perplexity=30.0, random_state=42).fit(X_df)

    # Validate embedding shape
    assert tsne.embedding_.shape == (100, 2)

