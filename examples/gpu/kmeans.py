import dpctl
import numpy as np

from sklearnex import config_context, patch_sklearn

patch_sklearn()

from sklearn.cluster import KMeans

# Create simple arrays
X = np.array(
    [[1.0, 2.0], [2.0, 2.0], [2.0, 3.0], [8.0, 7.0], [8.0, 8.0], [25.0, 80.0]],
    dtype=np.float32,
)

# Configure GPU context, and create simple model
with config_context(target_offload="gpu:0"):
    kmeans_model = KMeans(n_clusters=2).fit(X)
