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

# sklearnex kNN example for GPU offloading with DPNP ndarray:
#    python ./knn_bf_classification_dpnp_batch.py

import dpctl
import dpnp
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearnex.neighbors import KNeighborsClassifier

X, y = make_classification(
    n_samples=1000,
    n_features=4,
    n_informative=2,
    n_redundant=0,
    random_state=0,
    shuffle=False,
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Make sure that all DPNP ndarrays using the same device.
q = dpctl.SyclQueue("gpu")  # GPU

dpnp_X_train = dpnp.asarray(X_train, usm_type="device", sycl_queue=q)
dpnp_y_train = dpnp.asarray(y_train, usm_type="device", sycl_queue=q)
dpnp_X_test = dpnp.asarray(X_test, usm_type="device", sycl_queue=q)

knn_mdl = KNeighborsClassifier(
    algorithm="brute", n_neighbors=20, weights="uniform", p=2, metric="minkowski"
)
knn_mdl.fit(dpnp_X_train, dpnp_y_train)

y_predict = knn_mdl.predict(dpnp_X_test)

print("Brute Force Distributed kNN classification results:")
print("Ground truth (first 5 observations):\n{}".format(y_test[:5]))
print("Classification results (first 5 observations):\n{}".format(y_predict[:5]))
print("Accuracy (2 classes): {}\n".format(accuracy_score(y_test, y_predict.asnumpy())))
print("Are predicted results on GPU: {}".format(y_predict.sycl_device.is_gpu))
