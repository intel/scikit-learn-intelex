import numpy as np
import dpctl
from sklearnex import patch_sklearn, config_context
patch_sklearn()

from sklearn.ensemble import RandomForestRegressor

# Create simple arrays
X_train = np.array([[1., 2.], [2., 2.], [2., 3.],
            [8., 7.], [8., 8.], [25., 80.]], dtype=np.float32)
y_train = np.array([3., 4., 5., 15., 16., 105.], dtype=np.float32)
X_test = np.array([[0., 0.], [7., 8.]], dtype=np.float32)

# Configure GPU context, train simple model, and make inferences
with config_context(target_offload="gpu:0"):
    rfreg_model = RandomForestRegressor().fit(X_train, y_train)
    y_pred = rfreg_model.predict(X_test)
