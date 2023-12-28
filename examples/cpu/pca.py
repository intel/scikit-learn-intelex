import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.decomposition import PCA

# Create simple arrays
X = np.array([[1., 2.], [1., 9.], [5., 5.],
            [6., 4.], [8., 8.], [4., 4.]], dtype=np.float32)

# Fit simple model
pca_model = PCA(n_components=2).fit(X)
