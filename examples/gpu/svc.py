import numpy as np
import dpctl
from sklearnex import patch_sklearn, config_context

patch_sklearn()

from sklearn.svm import SVC

# Create simple arrays
X_train = np.array(
    [[1.0, 2.0], [1.0, 9.0], [5.0, 5.0], [6.0, 4.0], [8.0, 8.0], [4.0, 4.0]],
    dtype=np.float32,
)
y_train = np.array([0, 0, 1, 0, 1, 0], dtype=np.int32)
X_test = np.array([[9.0, 3.0], [6.0, 5.0]], dtype=np.float32)

# Configure GPU context, train simple model, and make inferences
with config_context(target_offload="gpu:0"):
    svc_model = SVC().fit(X_train, y_train)
    y_pred = svc_model.predict(X_test)
