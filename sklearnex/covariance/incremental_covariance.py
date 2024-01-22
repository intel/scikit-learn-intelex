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

import numpy as np
from sklearn.utils import check_array, gen_batches

from daal4py.sklearn._n_jobs_support import control_n_jobs
from onedal._device_offload import support_usm_ndarray
from onedal.covariance import (
    IncrementalEmpiricalCovariance as onedal_IncrementalEmpiricalCovariance,
)


@control_n_jobs(decorated_methods=["partial_fit"])
class IncrementalEmpiricalCovariance:
    """
    Incremental estimator for covariance.
    Allows to compute empirical covariance estimated by maximum
    likelihood method if data are splitted into batches.

    Parameters
    ----------
    batch_size : int, default=None
        The number of samples to use for each batch. Only used when calling
        ``fit``. If ``batch_size`` is ``None``, then ``batch_size``
        is inferred from the data and set to ``5 * n_features``, to provide a
        balance between approximation accuracy and memory consumption.

    Attributes
    ----------
    location_ : ndarray of shape (n_features,)
        Estimated location, i.e. the estimated mean.

    covariance_ : ndarray of shape (n_features, n_features)
        Estimated covariance matrix
    """

    _onedal_incremental_covariance = staticmethod(onedal_IncrementalEmpiricalCovariance)

    def __init__(self, batch_size=None):
        self._need_to_finalize = False  # If True then finalize compute should
        #      be called to obtain covariance_ or location_ from partial compute data
        self.batch_size = batch_size

    def _onedal_finalize_fit(self):
        assert hasattr(self, "_onedal_estimator")
        self._onedal_estimator.finalize_fit()
        self._need_to_finalize = False

    def _onedal_partial_fit(self, X, queue):
        onedal_params = {
            "method": "dense",
            "bias": True,
        }
        if not hasattr(self, "_onedal_estimator"):
            self._onedal_estimator = self._onedal_incremental_covariance(**onedal_params)
        self._onedal_estimator.partial_fit(X, queue)
        self._need_to_finalize = True

    @property
    def covariance_(self):
        if self._need_to_finalize:
            self._onedal_finalize_fit()
        return self._onedal_estimator.covariance_

    @property
    def location_(self):
        if self._need_to_finalize:
            self._onedal_finalize_fit()
        return self._onedal_estimator.location_

    @support_usm_ndarray()
    def partial_fit(self, X, queue=None):
        """
        Incremental fit with X. All of X is processed as a single batch.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = check_array(X, dtype=[np.float64, np.float32])
        self._onedal_partial_fit(X, queue)
        return self

    def fit(self, X, queue=None):
        """
        Fit the model with X, using minibatches of size batch_size.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        n_samples, n_features = X.shape
        if self.batch_size is None:
            batch_size_ = 5 * n_features
        else:
            batch_size_ = self.batch_size
        for batch in gen_batches(n_samples, batch_size_):
            X_batch = X[batch]
            self.partial_fit(X_batch, queue=queue)

        self._onedal_finalize_fit()
        return self
