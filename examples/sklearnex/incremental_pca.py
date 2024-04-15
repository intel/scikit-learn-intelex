# ==============================================================================
# Copyright 2024 Intel Corporation
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

from sklearnex.preview.decomposition import IncrementalPCA

incpca = IncrementalPCA()

# We do partial_fit for each batch and then print final result.
X_1 = np.array([[-1, -1], [-2, -1]])
result = incpca.partial_fit(X_1)

X_2 = np.array([[-3, -2], [1, 1]])
result = incpca.partial_fit(X_2)

X_3 = np.array([[2, 1], [3, 2]])
result = incpca.partial_fit(X_3)

X = np.concatenate((X_1, X_2, X_3))
transformed_X = incpca.transform(X)

print(f"Number of components:\n{result.n_components_}")
print(f"Singular values:\n{result.singular_values_}")
print(f"Means:\n{result.mean_}")
print(f"Explained variance:\n{result.explained_variance_}")
print(f"Explained variance ratio:\n{result.explained_variance_ratio_}")
print(f"Transformed data:\n{transformed_X}")

# We put the whole data to fit method, it is split automatically and then
# partial_fit is called for each batch.
incpca = IncrementalPCA(batch_size=3)
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
result = incpca.fit(X)
transformed_X = incpca.transform(X)

print(f"Number of components:\n{result.n_components_}")
print(f"Singular values:\n{result.singular_values_}")
print(f"Means:\n{result.mean_}")
print(f"Explained variance:\n{result.explained_variance_}")
print(f"Explained variance ratio:\n{result.explained_variance_ratio_}")
print(f"Transformed data:\n{transformed_X}")

# We call fit_transform method to get transformed data in single call.
incpca = IncrementalPCA(batch_size=3)
transformed_X = incpca.fit_transform(X)

print(f"Number of components:\n{incpca.n_components_}")
print(f"Singular values:\n{incpca.singular_values_}")
print(f"Means:\n{incpca.mean_}")
print(f"Explained variance:\n{incpca.explained_variance_}")
print(f"Explained variance ratio:\n{incpca.explained_variance_ratio_}")
print(f"Transformed data:\n{transformed_X}")
