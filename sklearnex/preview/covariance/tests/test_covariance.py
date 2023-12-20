# ===============================================================================
# Copyright 2023 Intel Corporation
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
    _convert_to_dataframe,
    get_dataframes_and_queues,
)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("macro_block", [None, 1024])
def test_sklearnex_import_covariance(dataframe, queue, macro_block):
    from sklearnex.preview.covariance import EmpiricalCovariance

    X = np.array([[0, 1], [0, 1]])
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    empcov = EmpiricalCovariance()
    if daal_check_version((2024, "P", 0)) and macro_block is not None:
        hparams = empcov.get_hyperparameters("fit")
        hparams.cpu_macro_block = macro_block
    result = empcov.fit(X)
    expected_covariance = np.array([[0, 0], [0, 0]])
    expected_means = np.array([0, 1])

    assert_allclose(expected_covariance, result.covariance_)
    assert_allclose(expected_means, result.location_)

    X = np.array([[1, 2], [3, 6]])
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    result = empcov.fit(X)
    expected_covariance = np.array([[1, 2], [2, 4]])
    expected_means = np.array([2, 4])

    assert_allclose(expected_covariance, result.covariance_)
    assert_allclose(expected_means, result.location_)
