import numpy as np
import dpctl
from sklearnex import patch_sklearn, config_context
patch_sklearn()

from sklearn.cluster import KMeans

# Create simple arrays
X = np.array([[1., 2.], [2., 2.], [2., 3.],
            [8., 7.], [8., 8.], [25., 80.]], dtype=np.float32)

# Configure GPU context, and create simple model
with config_context(target_offload="gpu:0"):
    kmeans_model = KMeans(n_clusters=2).fit(X)
