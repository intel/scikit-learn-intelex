# ===============================================================================
# Copyright 2022 Intel Corporation
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

import scipy.sparse as sp
from sklearn.utils.validation import _assert_all_finite as _sklearn_assert_all_finite
from onedal.utils.validation import _assert_all_finite as _onedal_assert_all_finite
from daal4py.sklearn._utils import sklearn_check_version

if sklearn_check_version("1.6"):
    from sklearn.utils.validation import validate_data as _sklearn_validate_data
    _finite_keyword = "ensure_all_finite"

else:
    from sklearn.base import BaseEstimator
    _sklearn_validate_data = BaseEstimator._validate_data
    _finite_keyword = "force_all_finite"



def validate_data(*args, **kwargs):
    # force finite check to not occur in sklearn, default is True
    force_all_finite = _finite_keyword not in kwargs or kwargs[_finite_keyword]
    kwargs[_finite_keyword] = False
    out = _sklearn_validate_data(*args, **kwargs)
    if force_all_finite:
        # run local finite check
        for arg in out:
            assert_all_finite(arg)
    return out


def assert_all_finite(
    X,
    *,
    allow_nan=False,
    input_name="",
):
    _assert_all_finite(
        X.data if sp.issparse(X) else X,
        allow_nan=allow_nan,
        input_name=input_name,
    )