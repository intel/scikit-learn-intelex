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

import numpy as np

from sklearnex.covariance import IncrementalEmpiricalCovariance

X = np.array([[0, 1], [0, 1]])
X_split = np.array_split(X, 2)

inccov = IncrementalEmpiricalCovariance()
for i in range(2):
    result = inccov.partial_fit(X_split[i])

print(f"Covariance matrix:\n{result.covariance_}")
print(f"Means:\n{result.location_}")
