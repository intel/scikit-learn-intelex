import dpctl
import numpy as np

from sklearnex import config_context, patch_sklearn

patch_sklearn()

from sklearn.decomposition import PCA

# Create simple arrays
X = np.array(
    [[1.0, 2.0], [1.0, 9.0], [5.0, 5.0], [6.0, 4.0], [8.0, 8.0], [4.0, 4.0]],
    dtype=np.float32,
)

# Configure GPU context, and create simple model
with config_context(target_offload="gpu:0"):
    pca_model = PCA(n_components=2).fit(X)
