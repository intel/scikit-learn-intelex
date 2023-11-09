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

from daal4py.sklearn._utils import daal_check_version, sklearn_check_version

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression as sklearn_logreg
from onedal.logistic_regression import LogisticRegression as onedal_logreg

from sklearnex import patch_sklearn

#from sklearnex import config_context

import dpctl
import time

queue = None

if dpctl.has_gpu_devices:
    queue = dpctl.SyclQueue("gpu")

'''
data = load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

X, y = X.astype(dtype), y.astype(dtype)
'''

prefix = "/export/users/anatolyv/scikit-learn_bench/data/" + "airline_"

X_train = np.load(prefix + "x_train.npy")
X_test = np.load(prefix + "x_test.npy")

y_train = np.load(prefix + "y_train.npy")
y_test = np.load(prefix + "y_test.npy")

dtype = np.float64
NUM_ITER = 20

X_train, y_train = X_train.astype(dtype), y_train.astype(dtype).reshape(-1)
X_test, y_test = X_test.astype(dtype), y_test.astype(dtype).reshape(-1)

'''
model1 = onedal_logreg(solver='newton-cg', max_iter=NUM_ITER)
tm1 = time.time()
model1.fit(X_train, y_train, queue)
tm2 = time.time()
print("oneDAL fit time", tm2 - tm1)
y_pred1 = model1.predict(X_test, queue)

print("oneDAL accuracy:", (y_pred1 == y_test).mean())

'''
patch_sklearn()
from sklearn.linear_model import LogisticRegression as sklearn_logreg

model2 = sklearn_logreg(solver='newton-cg', max_iter=NUM_ITER)
tm3 = time.time()
model2.fit(X_train, y_train)
tm4 = time.time()
print("sklearnex fit time", tm4 - tm3)
y_pred2 = model2.predict(X_test)
print("sklearnex accuracy:", (y_pred2 == y_test).mean())



#print(model1.n_iter_, model2.n_iter_)
#y_pred = model.predict(X_test)

#print("Accuracy:", (y_pred2 == y_test).mean())