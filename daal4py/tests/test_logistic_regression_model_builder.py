#===============================================================================
# Copyright 2020-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

import unittest
import daal4py as d4p
import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from daal4py.sklearn._utils import daal_check_version

from daal4py import _get__daal_link_version__ as dv
# First item is major version - 2021,
# second is minor+patch - 0110,
# third item is status - B
daal_version = (int(dv()[0:4]), dv()[10:11], int(dv()[4:8]))
reason = str(((2021, 'P', 1))) + " not supported in this library version "
reason += str(daal_version)


class LogRegModelBuilder(unittest.TestCase):
    @unittest.skipUnless(all([hasattr(d4p, 'logistic_regression_model_builder'),
                              daal_check_version(((2021, 'P', 1)))]), reason)
    def test_iris_with_intercept(self):
        X, y = load_iris(return_X_y=True)
        n_classes = 3
        clf = LogisticRegression(fit_intercept=True, max_iter=1000,
                                 random_state=0).fit(X, y)
        builder = d4p.logistic_regression_model_builder(n_classes=n_classes,
                                                        n_features=X.shape[1])
        builder.set_beta(clf.coef_, clf.intercept_)

        alg_pred = d4p.logistic_regression_prediction(nClasses=n_classes)

        pred_daal = alg_pred.compute(X, builder.model).prediction.flatten()
        pred_sklearn = clf.predict(X)
        self.assertTrue(np.allclose(pred_daal, pred_sklearn))

    @unittest.skipUnless(all([hasattr(d4p, 'logistic_regression_model_builder'),
                              daal_check_version(((2021, 'P', 1)))]), reason)
    def test_iris_without_intercept(self):
        X, y = load_iris(return_X_y=True)
        n_classes = 3
        clf = LogisticRegression(fit_intercept=False, max_iter=1000,
                                 random_state=0).fit(X, y)
        builder = d4p.logistic_regression_model_builder(n_classes=n_classes,
                                                        n_features=X.shape[1])
        builder.set_beta(clf.coef_, clf.intercept_)

        alg_pred = d4p.logistic_regression_prediction(nClasses=n_classes)

        pred_daal = alg_pred.compute(X, builder.model).prediction.flatten()
        pred_sklearn = clf.predict(X)
        self.assertTrue(np.allclose(pred_daal, pred_sklearn))

    @unittest.skipUnless(all([hasattr(d4p, 'logistic_regression_model_builder'),
                              daal_check_version(((2021, 'P', 1)))]), reason)
    def test_breast_cancer_with_intercept(self):
        X, y = load_breast_cancer(return_X_y=True)
        n_classes = 2
        clf = LogisticRegression(fit_intercept=True, max_iter=10000,
                                 random_state=0).fit(X, y)
        builder = d4p.logistic_regression_model_builder(n_classes=n_classes,
                                                        n_features=X.shape[1])
        builder.set_beta(clf.coef_, clf.intercept_)

        alg_pred = d4p.logistic_regression_prediction(nClasses=n_classes)

        pred_daal = alg_pred.compute(X, builder.model).prediction.flatten()
        pred_sklearn = clf.predict(X)
        self.assertTrue(np.allclose(pred_daal, pred_sklearn))

    @unittest.skipUnless(all([hasattr(d4p, 'logistic_regression_model_builder'),
                              daal_check_version(((2021, 'P', 1)))]), reason)
    def test_breast_cancer_without_intercept(self):
        X, y = load_breast_cancer(return_X_y=True)
        n_classes = 2
        clf = LogisticRegression(fit_intercept=False, max_iter=10000,
                                 random_state=0).fit(X, y)
        builder = d4p.logistic_regression_model_builder(n_classes=n_classes,
                                                        n_features=X.shape[1])
        builder.set_beta(clf.coef_, clf.intercept_)

        alg_pred = d4p.logistic_regression_prediction(nClasses=n_classes)

        pred_daal = alg_pred.compute(X, builder.model).prediction.flatten()
        pred_sklearn = clf.predict(X)
        self.assertTrue(np.allclose(pred_daal, pred_sklearn))


if __name__ == '__main__':
    unittest.main()
