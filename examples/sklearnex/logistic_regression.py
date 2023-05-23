from sklearnex import patch_sklearn, unpatch_sklearn
from daal4py.oneapi import sycl_context

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=10**5, n_features=50,
                           n_informative=40, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(solver='lbfgs', fit_intercept=True)
y_pred = model.fit(X_train, y_train).predict(X_test)
print("Sklearn LogisticRegression, accuracy on test:", accuracy_score(y_test, y_pred))

patch_sklearn(preview=True)

model_cpu = LogisticRegression(solver='lbfgs', fit_intercept=True)
y_pred_cpu = model_cpu.fit(X_train, y_train).predict(X_test)
print("CPU optimized version, accuracy on test:", accuracy_score(y_test, y_pred_cpu))
