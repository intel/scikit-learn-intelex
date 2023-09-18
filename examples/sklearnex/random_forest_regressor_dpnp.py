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

# sklearnex RF example for GPU offloading with DPNP ndarray:
#    python ./random_forest_regressor_dpnp_batch.py

import dpnp
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from sklearnex.preview.ensemble import RandomForestRegressor

sycl_device = "gpu:0"

X, y = make_regression(
    n_samples=1000, n_features=4, n_informative=2, random_state=0, shuffle=False
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

dpnp_X_train = dpnp.asarray(X_train, device=sycl_device)
dpnp_y_train = dpnp.asarray(y_train, device=sycl_device)
dpnp_X_test = dpnp.asarray(X_test, device=sycl_device)

rf = RandomForestRegressor(max_depth=2, random_state=0).fit(dpnp_X_train, dpnp_y_train)

pred = rf.predict(dpnp_X_test)

print("Random Forest regression results:")
print("Ground truth (first 5 observations):\n{}".format(y_test[:5]))
print("Regression results (first 5 observations):\n{}".format(pred[:5]))
print("Are predicted results on GPU: {}".format(pred.sycl_device.is_gpu))
