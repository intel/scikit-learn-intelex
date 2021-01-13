#===============================================================================
# Copyright 2014-2021 Intel Corporation
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

from sklearn.linear_model import LinearRegression as LinearRegression_original

from .._utils import get_patch_message

from ._linear_0_23 import (_fit_linear, _predict_linear)
import logging

class LinearRegression(LinearRegression_original):
    __doc__ = LinearRegression_original.__doc__

    def __init__(self, fit_intercept=True, normalize=False, copy_X=True,
                 n_jobs=None, positive=False):
        super(LinearRegression, self).__init__(
            fit_intercept=fit_intercept, normalize=normalize,
            copy_X=copy_X, n_jobs=n_jobs, positive=positive)


    def fit(self, X, y, sample_weight=None):
        if self.positive == True:
            logging.info("sklearn.linar_model.LinearRegression.fit: " + get_patch_message("sklearn"))
            return super(LinearRegression, self).fit(X, y=y, sample_weight=sample_weight)
        return _fit_linear(self, X, y, sample_weight=sample_weight)

    def predict(self, X):
        return _predict_linear(self, X)
