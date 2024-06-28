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

from sklearnex.model_selection import train_test_split

try:
    import dpctl
    import dpctl.tensor as dpt
    from dpctl import SyclQueue
    from mpi4py import MPI

    mpi_libs_available = True
    gpu_is_available = dpctl.has_gpu_devices()
except (ImportError, ModuleNotFoundError):
    mpi_libs_available = False

mpi_libs_and_gpu_available = mpi_libs_available and gpu_is_available


def get_local_tensor(full_data, data_parallel=True):
    """Splits data across ranks.

    Called on each rank to extract the subset of data assigned to that rank.

    Args:
        full_data (numpy array): The entire set of data
        data_parallel (bool): Whether or not to return dpctl array

    Returns:
        local_dpt_data (numpy or dpctl array): The subset of data used by the rank
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

    if not data_parallel:
        return local_data

    local_dpt_data = dpt.asarray(local_data, usm_type="device", sycl_queue=q)

    return local_dpt_data


def generate_regression_data(n_samples, n_features, random_state=42):
    # Generates regression data and divides between train and test
    X, y = make_regression(
        n_samples=n_samples, n_features=n_features, random_state=random_state
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
    return X_train, X_test, y_train, y_test


def generate_classification_data(n_samples, n_features, n_classes=2, random_state=42):
    # Generates classification data and divides between train and test
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=int(0.5 * n_classes + 1),
        random_state=random_state,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
    return X_train, X_test, y_train, y_test


def generate_statistic_data(n_samples, n_features random_state=42):
    # Generates statistical data and divides between train and test
    gen = np.random.default_rng(random_state)
    data = gen.uniform(low=-0.3, high=+0.7, size=(n_samples, n_features))
    return data


def generate_clustering_data(n_samples, n_features, centers=None, random_state=42):
    # Generates clustering data and divides between train and test
    X, y = make_blobs(
        n_samples=n_samples,
        centers=centers,
        n_features=n_features,
        random_state=random_state,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
    return X_train, X_test, y_train, y_test


def spmd_assert_allclose(spmd_result, batch_result, **kwargs):
    """Calls assert_allclose on spmd and batch results.

    Called on each rank to compare the spmd result specific to that rank and
    subset of batch result that corresponds to that rank.

    Args:
        spmd_result (numpy or dpctl array): The result for the subset of data on the rank the function is called from, computed by the spmd estimator
        batch_result (numpy array): The result for all data, computed by the batch estimator

    Raises:
        AssertionError: If all results are adequately close.
    """

    # extract chunk from batch result to match with local spmd result
    local_batch_result = get_local_tensor(batch_result, data_parallel=False)

    # convert to numpy if needed
    if not isinstance(spmd_result, np.ndarray):
        numpy_spmd_result = dpt.to_numpy(spmd_result)
    else:
        numpy_spmd_result = spmd_result

    assert_allclose(numpy_spmd_result, local_batch_result, **kwargs)


def assert_unordered_allclose(spmd_result, batch_result, localize=False):
    """Checks if rows in spmd and batch results are aligned, even if not in the same order.

    Called to verify correct unordered results are present. Useful to check KMeans centers
    or KNN neighbors, where order does not matter. Sorts inputs to handle unordering. Also
    capable of handling localization.

    Args:
        spmd_result (numpy or dpctl array): Result computed by the spmd estimator
        batch_result (numpy array): Result computed by batch estimator
        localize (bool): Whether of not spmd result is specific to the rank, in which case batch result needs to be localized

    Raises:
        AssertionError: If all results are adequately close.
    """

    sorted_spmd_result = np.sort(spmd_result, axis=1)
    if localize:
        local_batch_result = get_local_tensor(batch_result, data_parallel=False)
        sorted_batch_result = np.sort(local_batch_result, axis=1)
    else:
        sorted_batch_result = np.sort(batch_result, axis=1)

    assert_allclose(sorted_spmd_result, sorted_batch_result)


def assert_kmeans_labels_allclose(spmd_labels, batch_labels, spmd_centers, batch_centers):
    """Checks if labels for spmd and batch results are aligned, even cluster indices don't match.

    Called to verify labels are assigned the same way on spmd and batch. Uses raw labels (which
    may not match) to identify cluster center and ensure results match.

    Args:
        spmd_labels (numpy or dpctl array): The labels for the subset of data on the rank the function is called from, computed by the spmd estimator
        batch_labels (numpy array): The labels for all data, computed by the batch estimator
        spmd_centers (numpy or dpctl array): Centers computed by the spmd estimator
        batch_centers (numpy array): Centers computed by batch estimator

    Raises:
        AssertionError: If all results are adequately close.
    """

    if isinstance(spmd_labels, dpt.usm_ndarray):
        spmd_labels = dpt.to_numpy(spmd_labels)
    local_batch_labels = get_local_tensor(batch_labels, data_parallel=False)
    assert_allclose(spmd_centers[spmd_labels], batch_centers[local_batch_labels])
