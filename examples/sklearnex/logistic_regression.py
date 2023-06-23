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

from sklearnex import patch_sklearn, unpatch_sklearn
from daal4py.oneapi import sycl_context

from dpctl import SyclQueue
import dpctl.tensor as dpt

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

from daal4py.sklearn._utils import daal_check_version, sklearn_check_version


X, y = make_classification(n_samples=10**5, n_features=50,
                           n_informative=40, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(solver='lbfgs', fit_intercept=True)
y_pred = model.fit(X_train, y_train).predict(X_test)
print("Sklearn LogisticRegression, accuracy on test:", accuracy_score(y_test, y_pred))

patch_sklearn(preview=True)
from sklearn.linear_model import LogisticRegression

model_cpu = LogisticRegression(solver='lbfgs', fit_intercept=True)
y_pred_cpu = model_cpu.fit(X_train, y_train).predict(X_test)
print(
    "Sklearnex optimized version on CPU, accuracy on test:",
    accuracy_score(
        y_test,
        y_pred_cpu))

if (sklearn_check_version('1.1') and daal_check_version((2023, 'P', 200))):

    with sycl_context("gpu"):
        model_gpu = LogisticRegression(solver='lbfgs', fit_intercept=True)
        y_pred_gpu = model_gpu.fit(X_train, y_train).predict(X_test)
        print(
            "Sklearnex optimized version on GPU, accuracy on test:",
            accuracy_score(
                y_test,
                y_pred_gpu))

    ''' DPCTL tensor support
        Please note that regardless of where input
        dpctl tensors are stored CPU implementation will be used.
        To use gpu implementation please specify it with sycl_context.
    '''
    queue = SyclQueue("gpu")
    dpt_X_train = dpt.asarray(X_train, usm_type="device", sycl_queue=queue)
    dpt_X_test = dpt.asarray(X_test, usm_type="device", sycl_queue=queue)
    dpt_y_train = dpt.asarray(y_train, usm_type="device", sycl_queue=queue)

    model = LogisticRegression(solver='lbfgs', fit_intercept=True)
    model.fit(dpt_X_train, dpt_y_train)
    dpt_y_pred = model.predict(dpt_X_test)
    y_pred = dpt.to_numpy(dpt_y_pred)

    print("Sklearnex optimized version with dpctl tensors:",
          accuracy_score(y_test, y_pred_gpu))
