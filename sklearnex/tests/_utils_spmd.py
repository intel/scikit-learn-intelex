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

import numpy as np
from numpy.testing import assert_allclose
from sklearn.datasets import make_blobs, make_classification, make_regression
from sklearn.model_selection import train_test_split

from onedal.tests.utils._dataframes_support import _as_numpy

try:
    import dpctl
    from dpctl import SyclQueue
    from mpi4py import MPI

    mpi_libs_available = True
    gpu_is_available = dpctl.has_gpu_devices()
except (ImportError, ModuleNotFoundError):
    mpi_libs_available = False

_mpi_libs_and_gpu_available = mpi_libs_available and gpu_is_available


def _get_local_tensor(full_data):
    """Splits data across ranks.

    Called on each rank to extract the subset of data assigned to that rank.

    Args:
        full_data (numpy or dpctl array): The entire set of data

    Returns:
        local_data (numpy or dpctl array): The subset of data used by the rank
    """

    # create sycl queue and gather communicator details
    q = SyclQueue("gpu")
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # divide data across ranks and move to dpt tensor
    data_rows = full_data.shape[0]
    local_start = rank * data_rows // size
    local_end = (1 + rank) * data_rows // size
    local_data = full_data[local_start:local_end]

    return local_data


def _generate_regression_data(n_samples, n_features, dtype=np.float64, random_state=42):
    # Generates regression data and divides between train and test
    X, y = make_regression(
        n_samples=n_samples, n_features=n_features, random_state=random_state
    )
    X = X.astype(dtype)
    y = y.astype(dtype)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
    return X_train, X_test, y_train, y_test


def _spmd_assert_allclose(spmd_result, batch_result, **kwargs):
    """Calls assert_allclose on spmd and batch results.

    Called on each rank to compare the spmd result specific to that rank and
    subset of batch result that corresponds to that rank.

    Args:
        spmd_result (numpy or dpctl array): The result for the subset of data on the rank the function is called from, computed by the spmd estimator
        batch_result (numpy array): The result for all data, computed by the batch estimator

    Raises:
        AssertionError: If all results are not adequately close.
    """

    # extract chunk from batch result to match with local spmd result
    local_batch_result = _get_local_tensor(batch_result)

    assert_allclose(_as_numpy(spmd_result), _as_numpy(local_batch_result), **kwargs)
