import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.neighbors import KNeighborsClassifier

# Create simple arrays
X_train = np.array([[1., 2.], [1., 9.], [5., 5.],
            [6., 4.], [8., 8.], [4., 4.]], dtype=np.float32)
y_train = np.array([0, 0, 1, 0, 1, 0], dtype=np.int32)
X_test = np.array([[9., 3.], [6., 5.]], dtype=np.float32)

# Fit simple model and make inferences
knncls_model = KNeighborsClassifier(n_neighbors=2).fit(X_train, y_train)
y_pred = knncls_model.predict(X_test)
