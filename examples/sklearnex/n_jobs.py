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

# sklearnex support `n_jobs` parameter for all patched estimators
# even if original sklearn estimator doesn't

# Calling scikit-learn patch - this would enable acceleration on all enabled algorithms
from sklearnex import patch_sklearn

patch_sklearn()

# Remaining non modified scikit-learn code
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(
    n_samples=750, centers=centers, cluster_std=0.4, random_state=0
)

X = StandardScaler().fit_transform(X)

from sklearn.cluster import DBSCAN, KMeans

# DBSCAN originally supports `n_jobs`
db = DBSCAN(eps=0.3, min_samples=10, n_jobs=2).fit(X)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("DBSCAN - Estimated number of clusters: %d" % n_clusters_)
print("DBSCAN - Estimated number of noise points: %d" % n_noise_)

# KMeans doesn't originally support `n_jobs`
km = KMeans(n_clusters=2, init="k-means++", n_init=5, n_jobs=2).fit(X)
inertia_ = km.inertia_
n_iter_ = km.n_iter_

print("KMeans - Estimated number of iterations: %d" % n_iter_)
print("KMeans - Estimated inertia: %f" % inertia_)
