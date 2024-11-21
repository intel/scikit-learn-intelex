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

from daal4py.sklearn._utils import sklearn_check_version
from onedal.utils._array_api import _is_numpy_namespace
from onedal.utils.validation import _assert_all_finite as _onedal_assert_all_finite

from ._array_api import get_namespace

if sklearn_check_version("1.6"):
    from sklearn.utils.validation import validate_data as _sklearn_validate_data

    _finite_keyword = "ensure_all_finite"

else:
    from sklearn.base import BaseEstimator

    _sklearn_validate_data = BaseEstimator._validate_data
    _finite_keyword = "force_all_finite"


def _is_contiguous(X):
    # array_api does not have a `strides` or `flags` attribute for testing memory
    # order. When dlpack support is brought in for oneDAL, the dlpack python capsule
    # can then be inspected for strides and this must be updated. _is_contiguous is
    # therefore conservative in verifying attributes and does not support array_api.
    # This will block onedal_assert_all_finite from being used for array_api inputs.
    if hasattr(X, "flags") and X.flags["C_CONTIGUOUS"] or X.flags["F_CONTIGUOUS"]:
        return True
    return False


def _sycl_usm_assert_all_finite(X, xp, *, allow_nan=False, input_name=""):
    # This is a reproduction of code from sklearn.utils.validation necessary for
    # non-contiguous or non-fp32/fp64 dpctl inputs when sklearn version is <1.2 or
    # for non-contiguous or non-fp32/fp64 dpnp inputs, as these cannot be checked
    # for finiteness in sklearn nor onedal while preserving their object type.
    first_pass_isfinite = xp.isfinite(xp.sum(X))
    if first_pass_isfinite:
        return

    has_inf = xp.any(xp.isinf(X))
    has_nan_error = False if allow_nan else xp.any(xp.isnan(X))
    if has_inf or has_nan_error:
        type_err = "infinity" if allow_nan else "NaN, infinity"
        padded_input_name = input_name + " " if input_name else ""
        msg_err = f"Input {padded_input_name}contains {type_err}."
        raise ValueError(msg_err)


if sklearn_check_version("1.2"):

    def _general_assert_all_finite(
        X, xp, is_array_api_compliant, *, allow_nan=False, input_name=""
    ):
        if _is_numpy_namespace(xp) or is_array_api_compliant:
            _sklearn_assert_all_finite(X, allow_nan=allow_nan, input_name=input_name)
        elif "float" not in xp.dtype.name or "complex" not in xp.dtype.name:
            return
        # handle dpnp inputs
        _sycl_usm_assert_all_finite(X, xp, allow_nan=allow_nan, input_name=input_name)

else:

    def _general_assert_all_finite(
        X, xp, is_array_api_compliant, *, allow_nan=False, input_name=""
    ):

        if _is_numpy_namespace(xp):
            _sklearn_assert_all_finite(X, allow_nan, input_name=input_name)
        elif is_array_api_compliant and not xp.isdtype(
            X, ("real floating", "complex floating")
        ):
            return
        elif "float" not in xp.dtype.name or "complex" not in xp.dtype.name:
            return
        # handle dpctl and dpnp inputs
        _sycl_usm_assert_all_finite(X, xp, allow_nan, input_name=input_name)


def _assert_all_finite(
    X,
    *,
    allow_nan=False,
    input_name="",
):
    # unlike sklearnex, sklearn does not support sycl_usm_ndarrays by default
    # therefore a separate finite check implementation matching sklearn's
    # `_assert_all_finite` is necessary when the data is not float32 or float64 or
    # non-contiguous. The onedal assert_all_finite is only for float32 and float64
    # contiguous arrays.

    # size check is an initial match to daal4py for performance reasons, can be
    # optimized later
    xp, is_array_api_compliant = get_namespace(X)
    if X.size < 32768 or X.dtype not in [xp.float32, xp.float64] or not _is_contiguous(X):

        # all sycl_usm_ndarrays for sklearn < 1.2 and dpnp for sklearn > 1.2 are not
        # handled properly, it calls a separate function for an import-time sklearn
        # version check before possible hand-off to sklearn's _assert_all_finite or to
        # _assert_all_finite_core.
        _general_assert_all_finite(
            X, xp, is_array_api_compliant, allow_nan=allow_nan, input_name=input_name
        )
    else:
        _onedal_assert_all_finite(X, allow_nan=allow_nan, input_name=input_name)


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


def validate_data(
    _estimator,
    /,
    X="no_validation",
    y="no_validation",
    **kwargs,
):
    # force finite check to not occur in sklearn, default is True
    # `ensure_all_finite` is the most up-to-date keyword name in sklearn
    # _finite_keyword provides backward compatability for `force_all_finite`
    ensure_all_finite = (
        True if "ensure_all_finite" not in kwargs else kwargs.pop("ensure_all_finite")
    )
    kwargs[_finite_keyword] = False
    out = _sklearn_validate_data(
        _estimator,
        X=X,
        y=y,
        **kwargs,
    )
    if ensure_all_finite:
        # run local finite check
        allow_nan = ensure_all_finite == "allow-nan"
        arg = iter(out)
        if not isinstance(X, str) or X != "no_validation":
            assert_all_finite(next(arg), allow_nan=allow_nan, input_name="X")
        if y is not None or not isinstance(y, str) or y != "no_validation":
            assert_all_finite(next(arg), allow_nan=allow_nan, input_name="y")
    return out
