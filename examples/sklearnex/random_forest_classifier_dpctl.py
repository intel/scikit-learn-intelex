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

# sklearnex RF example for GPU offloading with DPCtl tensor:
#    python ./random_forest_classifier_dpctl_batch.py

import dpctl
import dpctl.tensor as dpt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sklearnex.preview.ensemble import RandomForestClassifier

# Make sure that all DPCtl tensors using the same device.
q = dpctl.SyclQueue("gpu")  # GPU

X, y = make_classification(
    n_samples=1000,
    n_features=4,
    n_informative=2,
    n_redundant=0,
    random_state=0,
    shuffle=False,
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

dpt_X_train = dpt.asarray(X_train, usm_type="device", sycl_queue=q)
dpt_y_train = dpt.asarray(y_train, usm_type="device", sycl_queue=q)
dpt_X_test = dpt.asarray(X_test, usm_type="device", sycl_queue=q)

rf = RandomForestClassifier(max_depth=2, random_state=0).fit(dpt_X_train, dpt_y_train)

pred = rf.predict(dpt_X_test)

print("Random Forest classification results:")
print("Ground truth (first 5 observations):\n{}".format(y_test[:5]))
print("Classification results (first 5 observations):\n{}".format(pred[:5]))
print("Are predicted results on GPU: {}".format(pred.sycl_device.is_gpu))
