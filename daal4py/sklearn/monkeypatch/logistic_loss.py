#
#*******************************************************************************
# Copyright 2014-2017 Intel Corporation
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
#******************************************************************************/

import numpy as np

import daal4py
from ..utils import (make2d, getFPType)

def _resultsToCompute_string(value=True, gradient=True, hessian=False):
    results_needed = []
    if value:
        results_needed.append('value')
    if gradient:
        results_needed.append('gradient')
    if hessian:
        results_needed.append('hessian')

    return '|'.join(results_needed)


def _daal4py_logistic_loss_extra_args(nClasses_unused, beta, X, y,
                                      l1=0.0, l2=0.0, fit_intercept=True,
                                      value=True, gradient=True, hessian=False):
    X = make2d(X)
    nSamples, nFeatures = X.shape

    y = make2d(y)
    beta = make2d(beta)
    n = X.shape[0]

    results_to_compute = _resultsToCompute_string(value=value, gradient=gradient, hessian=hessian)

    objective_function_algorithm_instance = daal4py.optimization_solver_logistic_loss(
        numberOfTerms = n,
        fptype = getFPType(X),
        method = 'defaultDense',
        interceptFlag = fit_intercept,
        penaltyL1 = l1 / n,
        penaltyL2 = l2 / n,
        resultsToCompute = results_to_compute
    )
    objective_function_algorithm_instance.setup(X, y, beta)

    return (objective_function_algorithm_instance, X, y, n)


def _daal4py_cross_entropy_loss_extra_args(nClasses, beta, X, y,
                                           l1=0.0, l2=0.0, fit_intercept=True,
                                           value=True, gradient=True, hessian=False):
    X = make2d(X)
    nSamples, nFeatures = X.shape
    y = make2d(y)
    beta = make2d(beta)
    n = X.shape[0]

    results_to_compute = _resultsToCompute_string(value=value, gradient=gradient, hessian=hessian)

    objective_function_algorithm_instance = daal4py.optimization_solver_cross_entropy_loss(
        nClasses = nClasses,
        numberOfTerms = n,
        fptype = getFPType(X),
        method = 'defaultDense',
        interceptFlag = fit_intercept,
        penaltyL1 = l1 / n,
        penaltyL2 = l2 / n,
        resultsToCompute = results_to_compute
    )
    objective_function_algorithm_instance.setup(X, y, beta)

    return (objective_function_algorithm_instance, X, y, n)


def _daal4py_loss_and_grad(beta, objF_instance, X, y, n):
    beta_ = make2d(beta)
    res = objF_instance.compute(X, y, beta_)
    gr = res.gradientIdx
    gr *= n
    v = res.valueIdx
    v *= n
    return (v, gr)
