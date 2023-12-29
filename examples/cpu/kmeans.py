import numpy as np
from sklearnex import patch_sklearn

patch_sklearn()

from sklearn.cluster import KMeans

# Create simple arrays
X = np.array(
    [[1.0, 2.0], [2.0, 2.0], [2.0, 3.0], [8.0, 7.0], [8.0, 8.0], [25.0, 80.0]],
    dtype=np.float32,
)

# Fit simple model
kmeans_model = KMeans(n_clusters=2).fit(X)
