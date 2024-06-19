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

import numpy
import pytest

from onedal.tests.utils._dataframes_support import (
    _convert_to_dataframe,
    get_dataframes_and_queues,
)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_onedal_ridge(dataframe, queue):
    from sklearn.linear_model import Ridge as Ridge_sklearn

    from sklearnex.preview.linear_model import Ridge as Ridge_sklearnex

    sample_count, feature_count = 10, 5
    alpha = 5.0

    sklearnex_model = Ridge_sklearnex(fit_intercept=True, alpha=alpha)
    sklearn_model = Ridge_sklearn(fit_intercept=True, alpha=alpha)

    X, y = numpy.random.rand(sample_count, feature_count), numpy.random.rand(sample_count)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)

    sklearnex_model.fit(X, y)
    sklearn_model.fit(X, y)
    numpy.testing.assert_allclose(sklearnex_model.coef_, sklearn_model.coef_)

    prediction_sklearnex = numpy.asarray(sklearnex_model.predict(X), dtype=numpy.float64)
    prediction_sklearn = numpy.asarray(sklearn_model.predict(X), dtype=numpy.float64)
    numpy.testing.assert_allclose(prediction_sklearnex, prediction_sklearn)
