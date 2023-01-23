#===============================================================================
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
#===============================================================================

import numpy as np
from onedal.datatypes import _check_array
from onedal import _backend

from ..common._policy import _get_policy
from ..datatypes._data_conversion import from_table, to_table


def _check_inputs(X, Y):
    def check_input(data):
        return _check_array(
            data,
            dtype=[np.float64, np.float32],
            force_all_finite=False
        )
    X = check_input(X)
    Y = X if Y is None else check_input(Y)
    fptype = 'float' if X.dtype is np.dtype('float32') else 'double'
    return X, Y, fptype


def _compute_cov(params, submodule, X, Y, queue):
    policy = _get_policy(queue, X, Y)
    X, Y = to_table(X, Y)
    result = submodule.compute(policy, params, X)
    return from_table(result.cov_matrix)


def covariance(X, Y=None, queue=None):
    """
    Compute the covariance matrix of X

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)

    Returns
    -------
    cov_matrix : ndarray of shape (n_features, n_features)
    """
    X, Y, fptype = _check_inputs(X, Y)
    return _compute_cov({'fptype': fptype, 'method': 'dense'},
                        _backend.covariance, X, Y, queue)
