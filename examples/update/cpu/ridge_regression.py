import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.linear_model import Ridge

# Create simple arrays
X_train = np.array([[1., 2.], [2., 2.], [2., 3.],
            [8., 7.], [8., 8.], [25., 80.]], dtype=np.float32)
y_train = np.array([3., 4., 5., 15., 16., 105.], dtype=np.float32)
X_test = np.array([[0., 0.], [7., 8.]], dtype=np.float32)

# Fit simple model, and make inferences
ridge_model = Ridge().fit(X_train, y_train)
y_pred = ridge_model.predict(X_test)
