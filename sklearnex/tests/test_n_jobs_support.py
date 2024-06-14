# ==============================================================================
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
# ==============================================================================

import inspect
import logging
from multiprocessing import cpu_count

import pytest
from sklearn.base import BaseEstimator
from sklearn.datasets import make_classification

from sklearnex.decomposition import PCA
from sklearnex.dispatcher import get_patch_map
from sklearnex.svm import SVC, NuSVC

ESTIMATORS = set(
    filter(
        lambda x: inspect.isclass(x) and issubclass(x, BaseEstimator),
        [value[0][0][2] for value in get_patch_map().values()],
    )
)

X, Y = make_classification(n_samples=40, n_features=4, random_state=42)


@pytest.mark.parametrize("estimator_class", ESTIMATORS)
@pytest.mark.parametrize("n_jobs", [None, -1, 1, 2])
def test_n_jobs_support(caplog, estimator_class, n_jobs):
    def check_estimator_doc(estimator):
        if estimator.__doc__ is not None:
            assert "n_jobs" in estimator.__doc__

    def check_n_jobs_entry_in_logs(caplog, function_name, n_jobs):
        for rec in caplog.records:
            if function_name in rec.message and "threads" in rec.message:
                expected_n_jobs = n_jobs if n_jobs > 0 else cpu_count() + 1 + n_jobs
                logging.info(f"{function_name}: setting {expected_n_jobs} threads")
                if f"{function_name}: setting {expected_n_jobs} threads" in rec.message:
                    return True
        # False if n_jobs is set and not found in logs
        return n_jobs is None

    def check_method(*args, method, caplog):
        method(*args)
        assert check_n_jobs_entry_in_logs(caplog, method.__name__, n_jobs)

    def check_methods_decoration(estimator):
        funcs = {
            i: getattr(estimator, i)
            for i in dir(estimator)
            if hasattr(estimator, i) and callable(getattr(estimator, i))
        }

        for func_name, func in funcs.items():
            assert hasattr(func, "__onedal_n_jobs_decorated__") == (
                func_name in estimator._n_jobs_supported_onedal_methods
            ), f"{estimator}.{func_name} n_jobs decoration does not match {estimator} n_jobs supported methods"

    caplog.set_level(logging.DEBUG, logger="sklearnex")
    estimator_kwargs = {"n_jobs": n_jobs}
    # by default, [Nu]SVC.predict_proba is restricted by @available_if decorator
    if estimator_class in [SVC, NuSVC]:
        estimator_kwargs["probability"] = True
    # explicitly request oneDAL's PCA-Covariance algorithm
    if estimator_class == PCA:
        estimator_kwargs["svd_solver"] = "covariance_eigh"
    estimator_instance = estimator_class(**estimator_kwargs)
    # check `n_jobs` parameter doc entry
    check_estimator_doc(estimator_class)
    check_estimator_doc(estimator_instance)
    # check `n_jobs` log entry for supported methods
    # `fit` call is required before other methods
    check_method(X, Y, method=estimator_instance.fit, caplog=caplog)
    for method_name in estimator_instance._n_jobs_supported_onedal_methods:
        if method_name == "fit":
            continue
        method = getattr(estimator_instance, method_name)
        argdict = inspect.signature(method).parameters
        argnum = len(
            [i for i in argdict if argdict[i].default == inspect.Parameter.empty]
        )
        if argnum == 0:
            check_method(method=method, caplog=caplog)
        elif argnum == 1:
            check_method(X, method=method, caplog=caplog)
        else:
            check_method(X, Y, method=method, caplog=caplog)
    # check if correct methods were decorated
    check_methods_decoration(estimator_class)
    check_methods_decoration(estimator_instance)
