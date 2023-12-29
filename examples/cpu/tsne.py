import numpy as np
from sklearnex import patch_sklearn

patch_sklearn()

from sklearn.manifold import TSNE

# Create simple arrays
X = np.array(
    [[1.0, 2.0], [1.0, 9.0], [5.0, 5.0], [6.0, 4.0], [8.0, 8.0], [4.0, 4.0]],
    dtype=np.float32,
)

# Fit simple model
tsne_model = TSNE(n_components=2, perplexity=3).fit(X)
