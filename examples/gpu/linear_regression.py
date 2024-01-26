import dpctl
import numpy as np

from sklearnex import config_context, patch_sklearn

patch_sklearn()

from sklearn.linear_model import LinearRegression

# Create simple arrays
X_train = np.array(
    [[1.0, 2.0], [2.0, 2.0], [2.0, 3.0], [8.0, 7.0], [8.0, 8.0], [25.0, 80.0]],
    dtype=np.float32,
)
y_train = np.array([3.0, 4.0, 5.0, 15.0, 16.0, 105.0], dtype=np.float32)
X_test = np.array([[0.0, 0.0], [7.0, 8.0]], dtype=np.float32)

# Configure GPU context, train simple model, and make inferences
with config_context(target_offload="gpu:0"):
    linreg_model = LinearRegression().fit(X_train, y_train)
    y_pred = linreg_model.predict(X_test)
