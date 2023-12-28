import numpy as np
import dpctl
from sklearnex import patch_sklearn, config_context
patch_sklearn()

from sklearn.decomposition import PCA

# Create simple arrays
X = np.array([[1., 2.], [1., 9.], [5., 5.],
            [6., 4.], [8., 8.], [4., 4.]], dtype=np.float32)

# Configure GPU context, and create simple model
with config_context(target_offload="gpu:0"):
    pca_model = PCA(n_components=2).fit(X)
