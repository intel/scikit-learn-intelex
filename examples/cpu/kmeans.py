import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.cluster import KMeans

# Create simple arrays
X = np.array([[1., 2.], [2., 2.], [2., 3.],
            [8., 7.], [8., 8.], [25., 80.]], dtype=np.float32)

# Fit simple model
kmeans_model = KMeans(n_clusters=2).fit(X)
