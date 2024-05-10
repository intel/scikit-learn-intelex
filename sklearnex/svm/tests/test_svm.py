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
import pytest
from numpy.testing import assert_allclose
from sklearn.utils.estimator_checks import check_estimator

from onedal.tests.utils._dataframes_support import (
    _as_numpy,
    _convert_to_dataframe,
    get_dataframes_and_queues,
)
from sklearnex.svm import SVC, SVR


def _replace_and_save(md, fns, replacing_fn):
    saved = dict()
    for check_f in fns:
        try:
            fn = getattr(md, check_f)
            setattr(md, check_f, replacing_fn)
            saved[check_f] = fn
        except RuntimeError:
            pass
    return saved


def _restore_from_saved(md, saved_dict):
    for check_f in saved_dict:
        setattr(md, check_f, saved_dict[check_f])


@pytest.mark.parametrize("estimator", [SVC, SVR])
def test_estimator(estimator):
    def dummy(*args, **kwargs):
        pass

    md = sklearn.utils.estimator_checks
    saved = _replace_and_save(
        md,
        [
            "check_sample_weights_invariance",  # Max absolute difference: 0.0008
            "check_estimators_fit_returns_self",  # ValueError: empty metadata
            "check_classifiers_train",  # assert y_pred.shape == (n_samples,)
            "check_estimators_unfitted",  # Call 'fit' with appropriate arguments
        ],
        dummy,
    )
    check_estimator(estimator())
    _restore_from_saved(md, saved)


# TODO:
# investigate failure for `dpnp.ndarrays` and `dpctl.tensors` on `GPU`
@pytest.mark.parametrize(
    "dataframe,queue", get_dataframes_and_queues(device_filter_="cpu")
)
def test_sklearnex_import_svc(dataframe, queue):
    from sklearnex.svm import SVC

    X = np.array([[-2, -1], [-1, -1], [-1, -2], [+1, +1], [+1, +2], [+2, +1]])
    y = np.array([1, 1, 1, 2, 2, 2])
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    svc = SVC(kernel="linear").fit(X, y)
    assert "daal4py" in svc.__module__ or "sklearnex" in svc.__module__
    assert_allclose(_as_numpy(svc.dual_coef_), [[-0.25, 0.25]])
    assert_allclose(_as_numpy(svc.support_), [1, 3])


# TODO:
# investigate failure for `dpnp.ndarrays` and `dpctl.tensors` on `GPU`
@pytest.mark.parametrize(
    "dataframe,queue", get_dataframes_and_queues(device_filter_="cpu")
)
def test_sklearnex_import_nusvc(dataframe, queue):
    from sklearnex.svm import NuSVC

    X = np.array([[-2, -1], [-1, -1], [-1, -2], [+1, +1], [+1, +2], [+2, +1]])
    y = np.array([1, 1, 1, 2, 2, 2])
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    svc = NuSVC(kernel="linear").fit(X, y)
    assert "daal4py" in svc.__module__ or "sklearnex" in svc.__module__
    assert_allclose(
        _as_numpy(svc.dual_coef_), [[-0.04761905, -0.0952381, 0.0952381, 0.04761905]]
    )
    assert_allclose(_as_numpy(svc.support_), [0, 1, 3, 4])


# TODO:
# investigate failure for `dpnp.ndarrays` and `dpctl.tensors` on `GPU`
@pytest.mark.parametrize(
    "dataframe,queue", get_dataframes_and_queues(device_filter_="cpu")
)
def test_sklearnex_import_svr(dataframe, queue):
    from sklearnex.svm import SVR

    X = np.array([[-2, -1], [-1, -1], [-1, -2], [+1, +1], [+1, +2], [+2, +1]])
    y = np.array([1, 1, 1, 2, 2, 2])
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    svc = SVR(kernel="linear").fit(X, y)
    assert "daal4py" in svc.__module__ or "sklearnex" in svc.__module__
    assert_allclose(_as_numpy(svc.dual_coef_), [[-0.1, 0.1]])
    assert_allclose(_as_numpy(svc.support_), [1, 3])


# TODO:
# investigate failure for `dpnp.ndarrays` and `dpctl.tensors` on `GPU`
@pytest.mark.parametrize(
    "dataframe,queue", get_dataframes_and_queues(device_filter_="cpu")
)
def test_sklearnex_import_nusvr(dataframe, queue):
    from sklearnex.svm import NuSVR

    X = np.array([[-2, -1], [-1, -1], [-1, -2], [+1, +1], [+1, +2], [+2, +1]])
    y = np.array([1, 1, 1, 2, 2, 2])
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    svc = NuSVR(kernel="linear", nu=0.9).fit(X, y)
    assert "daal4py" in svc.__module__ or "sklearnex" in svc.__module__
    assert_allclose(
        _as_numpy(svc.dual_coef_), [[-1.0, 0.611111, 1.0, -0.611111]], rtol=1e-3
    )
    assert_allclose(_as_numpy(svc.support_), [1, 2, 3, 5])
