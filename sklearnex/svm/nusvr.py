# ==============================================================================
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
# ==============================================================================

from sklearn.svm import NuSVR as sklearn_NuSVR
from sklearn.utils.validation import _deprecate_positional_args

from daal4py.sklearn._n_jobs_support import control_n_jobs
from daal4py.sklearn._utils import sklearn_check_version
from onedal.svm import NuSVR as onedal_NuSVR

from .._device_offload import dispatch, wrap_output_data
from ._common import BaseSVR


@control_n_jobs(decorated_methods=["fit", "predict"])
class NuSVR(sklearn_NuSVR, BaseSVR):
    __doc__ = sklearn_NuSVR.__doc__

    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {**sklearn_NuSVR._parameter_constraints}

    @_deprecate_positional_args
    def __init__(
        self,
        *,
        nu=0.5,
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0.0,
        shrinking=True,
        tol=1e-3,
        cache_size=200,
        verbose=False,
        max_iter=-1,
    ):
        super().__init__(
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            C=C,
            nu=nu,
            shrinking=shrinking,
            cache_size=cache_size,
            verbose=verbose,
            max_iter=max_iter,
        )

    def fit(self, X, y, sample_weight=None):
        if sklearn_check_version("1.2"):
            self._validate_params()
        if sklearn_check_version("1.0"):
            self._check_feature_names(X, reset=True)
        dispatch(
            self,
            "fit",
            {
                "onedal": self.__class__._onedal_fit,
                "sklearn": sklearn_NuSVR.fit,
            },
            X,
            y,
            sample_weight,
        )
        return self

    @wrap_output_data
    def predict(self, X):
        if sklearn_check_version("1.0"):
            self._check_feature_names(X, reset=False)
        return dispatch(
            self,
            "predict",
            {
                "onedal": self.__class__._onedal_predict,
                "sklearn": sklearn_NuSVR.predict,
            },
            X,
        )

    def _onedal_fit(self, X, y, sample_weight=None, queue=None):
        onedal_params = {
            "C": self.C,
            "nu": self.nu,
            "kernel": self.kernel,
            "degree": self.degree,
            "gamma": self.gamma,
            "coef0": self.coef0,
            "tol": self.tol,
            "shrinking": self.shrinking,
            "cache_size": self.cache_size,
            "max_iter": self.max_iter,
        }

        self._onedal_estimator = onedal_NuSVR(**onedal_params)
        self._onedal_estimator.fit(X, y, sample_weight, queue=queue)
        self._save_attributes()

    def _onedal_predict(self, X, queue=None):
        return self._onedal_estimator.predict(X, queue=queue)

    fit.__doc__ = sklearn_NuSVR.fit.__doc__
    predict.__doc__ = sklearn_NuSVR.predict.__doc__
