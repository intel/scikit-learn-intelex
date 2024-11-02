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

import scipy.sparse as sp

from onedal import _backend
from onedal.common._policy import _get_policy
from onedal.datatypes import _convert_to_supported, to_table


def _assert_all_finite(X, allow_nan=False, input_name=""):
    # NOTE: This function does not respond to target_offload, as the memory movement
    # is likely to cause a significant reduction in performance
    policy = _get_policy(None, X)
    X_table = to_table(_convert_to_supported(policy, X))
    if not _backend.finiteness_checker.compute(
        policy, {"allow_nan": allow_nan}, X_table
    ).finite:
        type_err = "infinity" if allow_nan else "NaN, infinity"
        padded_input_name = input_name + " " if input_name else ""
        msg_err = f"Input {padded_input_name}contains {type_err}."
        raise ValueError(msg_err)


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
