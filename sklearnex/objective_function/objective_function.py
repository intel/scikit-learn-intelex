# ===============================================================================
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
# ===============================================================================

from .._device_offload import dispatch, wrap_output_data

from daal4py.sklearn._utils import daal_check_version, sklearn_check_version
import logging
from scipy import sparse as sp

if daal_check_version((2023, 'P', 200)) and sklearn_check_version('1.1'):
    import numpy as np

        
    from daal4py.sklearn._utils import (get_dtype, make2d)
    from sklearn.linear_model._linear_loss import LinearModelLoss as sklearn_LinearModelLoss
    from sklearn._loss.loss import HalfBinomialLoss

    from onedal.objective_function import LogisticLoss as onedal_LogisticLoss

    class LinearModelLoss(sklearn_LinearModelLoss):

        def __init__(self, base_loss, fit_intercept=True):
            super().__init__(base_loss, fit_intercept)
            if (type(base_loss) == HalfBinomialLoss):
                self.onedal_base_loss = onedal_LogisticLoss(fit_intercept=fit_intercept)
    

        def _onedal_ready(self, coef, X, y, sample_weight, l2_reg_strength, n_threads, raw_prediction):
            if sp.issparse(coef) or sp.issparse(X) or sp.issparse(y):
                raise ValueError(
                    "sparse data is not supported."
                )
            if (l2_reg_strength < 0.0):
                raise ValueError("l2_reg_strength should be greater than zero")
            if (n_threads != 1):
                raise ValueError("multithreading is not supported")
            if (sample_weight != None):
                raise ValueError("custom sample weigths are not supported")
            if (raw_prediction != None):
                raise ValueError("raw predictions are not supported")
            
            return True

        def _onedal_cpu_supported(self, method_name, *data):
            return True

        def _onedal_gpu_supported(self, method_name, *data):
            return True


        def onedal_loss(
            self, *args, **kwargs):
            return self.onedal_base_loss.loss(*args, **kwargs)

        def onedal_gradient(
            self, *args, **kwargs):
            return self.onedal_base_loss.gradient(*args, **kwargs)
        
        def onedal_loss_gradient(
            self, *args, **kwargs):
            return self.onedal_base_loss.loss_gradient(*args, **kwargs)

        def onedal_gradient_hessian(
            self, *args, **kwargs):
            return self.onedal_base_loss.gradient_hessian(*args, **kwargs)

        def onedal_gradient_hessian_product(
            self, *args, **kwargs):
            return self.onedal_base_loss.gradient_hessian_product(*args, **kwargs)

        @wrap_output_data
        def loss(
            self,
            coef,
            X,
            y,
            sample_weight=None,
            l2_reg_strength=0.0,
            n_threads=1,
            raw_prediction=None,):
            if type(self.base_loss) == HalfBinomialLoss:
                return dispatch(self, 'sklearn.linear_model._linear_loss.LinearModelLoss.loss', {
                        'onedal': self.__class__.onedal_loss,
                        'sklearn': sklearn_LinearModelLoss.loss,
                    }, coef, X, y, sample_weight, l2_reg_strength, n_threads, raw_prediction)
            else:
                return super().loss(coef, X, y, sample_weight, l2_reg_strength, n_threads, raw_prediction)


        @wrap_output_data
        def loss_gradient(
            self,
            coef,
            X,
            y,
            sample_weight=None,
            l2_reg_strength=0.0,
            n_threads=1,
            raw_prediction=None,):
            if type(self.base_loss == HalfBinomialLoss):
                return dispatch(self, 'sklearn.linear_model._linear_loss.LinearModelLoss.loss_gradient', {
                        'onedal': self.__class__.onedal_loss_gradient,
                        'sklearn': sklearn_LinearModelLoss.loss_gradient,
                    }, coef, X, y, sample_weight, l2_reg_strength, n_threads, raw_prediction)
            else:
                return super().loss_gradient(coef, X, y, sample_weight, l2_reg_strength, n_threads, raw_prediction)


        @wrap_output_data
        def gradient(
            self,
            coef,
            X,
            y,
            sample_weight=None,
            l2_reg_strength=0.0,
            n_threads=1,
            raw_prediction=None,):

            if type(self.base_loss) == HalfBinomialLoss:
                return dispatch(self, 'sklearn.linear_model._linear_loss.LinearModelLoss.gradient', {
                        'onedal': self.__class__.onedal_gradient,
                        'sklearn': sklearn_LinearModelLoss.gradient,
                    }, coef, X, y, sample_weight, l2_reg_strength, n_threads, raw_prediction)
            else:
                return super().gradient(coef, X, y, sample_weight, l2_reg_strength, n_threads, raw_prediction)




        @wrap_output_data
        def gradient_hessian(
            self,
            coef,
            X,
            y,
            sample_weight=None,
            l2_reg_strength=0.0,
            n_threads=1,
            raw_prediction=None,):

            if type(self.base_loss) == HalfBinomialLoss:
                return dispatch(self, 'sklearn.linear_model._linear_loss.LinearModelLoss.gradient_hessian', {
                        'onedal': self.__class__.onedal_gradient_hessian,
                        'sklearn': sklearn_LinearModelLoss.gradient_hessian,
                    }, coef, X, y, sample_weight, l2_reg_strength, n_threads, raw_prediction)
            else:
                return super().gradient_hessian(coef, X, y, sample_weight, l2_reg_strength, n_threads, raw_prediction)


        @wrap_output_data
        def gradient_hessian_product(
            self,
            coef,
            X,
            y,
            sample_weight=None,
            l2_reg_strength=0.0,
            n_threads=1,
            raw_prediction=None,):

            if type(self.base_loss) == HalfBinomialLoss:
                return dispatch(self, 'sklearn.linear_model._linear_loss.LinearModelLoss.gradient_hessian_product', {
                        'onedal': self.__class__.onedal_gradient_hessian_product,
                        'sklearn': sklearn_LinearModelLoss.gradient_hessian_product,
                    }, coef, X, y, sample_weight, l2_reg_strength, n_threads, raw_prediction)
            else:
                return super().gradient_hessian_product(coef, X, y, sample_weight, l2_reg_strength, n_threads, raw_prediction)
else:
    logging.warning('LogisticLoss requires oneDAL version >= 2023.2 '
                    'but it was not found')
