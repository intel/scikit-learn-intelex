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
import pytest
from numpy.testing import assert_allclose

from onedal.tests.utils._spmd_support import (
    generate_classification_data,
    get_local_tensor,
    mpi_libs_and_gpu_available,
    spmd_assert_allclose,
)


@pytest.mark.skipif(
    not mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.mpi
def test_logistic_spmd_gold():
    # Import spmd and batch algo
    from sklearnex.linear_model import LogisticRegression as LogisticRegression_Batch
    from sklearnex.spmd.linear_model import LogisticRegression as LogisticRegression_SPMD

    # Create gold data and process into dpt
    X_train = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 2.0],
            [2.0, 0.0],
            [1.0, 1.0],
            [0.0, -1.0],
            [-1.0, 0.0],
            [-1.0, -1.0],
        ]
    )
    y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0])
    X_test = np.array(
        [
            [1.0, -1.0],
            [-1.0, 1.0],
            [0.0, 1.0],
            [10.0, -10.0],
        ]
    )

    local_dpt_X_train = get_local_tensor(X_train)
    local_dpt_y_train = get_local_tensor(y_train)
    local_dpt_X_test = get_local_tensor(X_test)

    # ensure trained model of batch algo matches spmd
    spmd_model = LogisticRegression_SPMD(random_state=0, solver="newton-cg").fit(
        local_dpt_X_train, local_dpt_y_train
    )
    batch_model = LogisticRegression_Batch(random_state=0, solver="newton-cg").fit(
        X_train, y_train
    )

    assert_allclose(spmd_model.coef_, batch_model.coef_, rtol=5e-4)
    assert_allclose(spmd_model.intercept_, batch_model.intercept_, rtol=5e-4)

    # ensure predictions of batch algo match spmd
    spmd_result = spmd_model.predict(local_dpt_X_test)
    batch_result = batch_model.predict(X_test)

    spmd_assert_allclose(spmd_result, batch_result)


# parametrize max_iter, C, tol
@pytest.mark.skipif(
    not mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.parametrize("n_samples", [100, 10000])
@pytest.mark.parametrize("n_features", [10, 100])
@pytest.mark.parametrize("C", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("tol", [1e-2, 1e-4])
@pytest.mark.mpi
def test_logistic_spmd_synthetic(n_samples, n_features, C, tol):
    # Import spmd and batch algo
    from sklearnex.linear_model import LogisticRegression as LogisticRegression_Batch
    from sklearnex.spmd.linear_model import LogisticRegression as LogisticRegression_SPMD

    # Generate data and process into dpt
    X_train, X_test, y_train, _ = generate_classification_data(n_samples, n_features)

    local_dpt_X_train = get_local_tensor(X_train)
    local_dpt_y_train = get_local_tensor(y_train)
    local_dpt_X_test = get_local_tensor(X_test)

    # ensure trained model of batch algo matches spmd
    spmd_model = LogisticRegression_SPMD(
        random_state=0, solver="newton-cg", C=C, tol=tol
    ).fit(local_dpt_X_train, local_dpt_y_train)
    batch_model = LogisticRegression_Batch(
        random_state=0, solver="newton-cg", C=C, tol=tol
    ).fit(X_train, y_train)

    # Not deterministic so no n_iter_ check and relatively flexible coef_ check
    if n_samples > 10 * n_features:
        assert_allclose(spmd_model.coef_, batch_model.coef_, rtol=tol)
        assert_allclose(spmd_model.intercept_, batch_model.intercept_, rtol=tol)

    # ensure predictions of batch algo match spmd
    spmd_result = spmd_model.predict(local_dpt_X_test)
    batch_result = batch_model.predict(X_test)

    spmd_assert_allclose(spmd_result, batch_result)
