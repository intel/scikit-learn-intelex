# ==============================================================================
# Copyright 2020 Intel Corporation
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
# ==============================================================================

import unittest

import catboost as cb
import lightgbm as lgbm
import numpy as np
import xgboost as xgb
from sklearn.datasets import (
    load_breast_cancer,
    load_iris,
    make_classification,
    make_regression,
)
from sklearn.linear_model import LogisticRegression

import daal4py as d4p


class LogRegModelBuilder(unittest.TestCase):
    def test_iris_with_intercept(self):
        X, y = load_iris(return_X_y=True)
        n_classes = 3
        clf = LogisticRegression(fit_intercept=True, max_iter=1000, random_state=0).fit(
            X, y
        )
        builder = d4p.logistic_regression_model_builder(
            n_classes=n_classes, n_features=X.shape[1]
        )
        builder.set_beta(clf.coef_, clf.intercept_)

        alg_pred = d4p.logistic_regression_prediction(nClasses=n_classes)

        pred_daal = alg_pred.compute(X, builder.model).prediction.flatten()
        pred_sklearn = clf.predict(X)
        self.assertTrue(np.allclose(pred_daal, pred_sklearn))

    def test_iris_without_intercept(self):
        X, y = load_iris(return_X_y=True)
        n_classes = 3
        clf = LogisticRegression(fit_intercept=False, max_iter=1000, random_state=0).fit(
            X, y
        )
        builder = d4p.logistic_regression_model_builder(
            n_classes=n_classes, n_features=X.shape[1]
        )
        builder.set_beta(clf.coef_, clf.intercept_)

        alg_pred = d4p.logistic_regression_prediction(nClasses=n_classes)

        pred_daal = alg_pred.compute(X, builder.model).prediction.flatten()
        pred_sklearn = clf.predict(X)
        self.assertTrue(np.allclose(pred_daal, pred_sklearn))

    def test_breast_cancer_with_intercept(self):
        X, y = load_breast_cancer(return_X_y=True)
        n_classes = 2
        clf = LogisticRegression(fit_intercept=True, max_iter=10000, random_state=0).fit(
            X, y
        )
        builder = d4p.logistic_regression_model_builder(
            n_classes=n_classes, n_features=X.shape[1]
        )
        builder.set_beta(clf.coef_, clf.intercept_)

        alg_pred = d4p.logistic_regression_prediction(nClasses=n_classes)

        pred_daal = alg_pred.compute(X, builder.model).prediction.flatten()
        pred_sklearn = clf.predict(X)
        self.assertTrue(np.allclose(pred_daal, pred_sklearn))

    def test_breast_cancer_without_intercept(self):
        X, y = load_breast_cancer(return_X_y=True)
        n_classes = 2
        clf = LogisticRegression(fit_intercept=False, max_iter=10000, random_state=0).fit(
            X, y
        )
        builder = d4p.logistic_regression_model_builder(
            n_classes=n_classes, n_features=X.shape[1]
        )
        builder.set_beta(clf.coef_, clf.intercept_)

        alg_pred = d4p.logistic_regression_prediction(nClasses=n_classes)

        pred_daal = alg_pred.compute(X, builder.model).prediction.flatten()
        pred_sklearn = clf.predict(X)
        self.assertTrue(np.allclose(pred_daal, pred_sklearn))


class XGBoostRegressionModelBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = make_regression(n_samples=2, n_features=10, random_state=42)
        cls.X_nan = np.array([np.nan] * 20, dtype=np.float32).reshape(2, 10)
        cls.xgb_model = xgb.XGBRegressor(max_depth=5, n_estimators=50, random_state=42)
        cls.xgb_model.fit(cls.X, cls.y)

    def test_model_conversion(self):
        m = d4p.mb.convert_model(self.xgb_model.get_booster())
        self.assertEqual(m.n_classes_, 0)
        self.assertEqual(m.n_features_in_, 10)
        self.assertTrue(m._is_regression)

    def test_model_predict(self):
        m = d4p.mb.convert_model(self.xgb_model.get_booster())
        d4p_pred = m.predict(self.X)
        xgboost_pred = self.xgb_model.predict(self.X)
        self.assertTrue(
            np.allclose(d4p_pred, xgboost_pred, atol=1e-7),
            f"d4p and reference prediction are different (d4p - ref) = {d4p_pred - xgboost_pred}",
        )

    def test_missing_value_support(self):
        m = d4p.mb.convert_model(self.xgb_model.get_booster())
        d4p_pred = m.predict(self.X_nan)
        xgboost_pred = self.xgb_model.predict(self.X_nan)
        self.assertTrue(
            np.allclose(d4p_pred, xgboost_pred, atol=1e-7),
            f"d4p and reference missing value prediction different (d4p - ref) = {d4p_pred - xgboost_pred}",
        )

    def test_model_predict_shap_contribs(self):
        booster = self.xgb_model.get_booster()
        m = d4p.mb.convert_model(booster)
        d4p_pred = m.predict(self.X, pred_contribs=True)
        xgboost_pred = booster.predict(
            xgb.DMatrix(self.X),
            pred_contribs=True,
            approx_contribs=False,
            validate_features=False,
        )
        self.assertTrue(
            d4p_pred.shape == xgboost_pred.shape,
            f"d4p and reference SHAP contribution shape is different {d4p_pred.shape} != {xgboost_pred.shape}",
        )
        self.assertTrue(
            np.allclose(d4p_pred, xgboost_pred, atol=1e-7),
            f"d4p and reference SHAP contribution prediction are different (d4p - ref) = {d4p_pred - xgboost_pred}",
        )

    def test_model_predict_shap_interactions(self):
        booster = self.xgb_model.get_booster()
        m = d4p.mb.convert_model(booster)
        d4p_pred = m.predict(self.X, pred_interactions=True)
        xgboost_pred = booster.predict(
            xgb.DMatrix(self.X),
            pred_interactions=True,
            approx_contribs=False,
            validate_features=False,
        )
        self.assertTrue(
            d4p_pred.shape == xgboost_pred.shape,
            f"d4p and reference SHAP interaction shape is different {d4p_pred.shape} != {xgboost_pred.shape}",
        )
        self.assertTrue(
            np.allclose(d4p_pred, xgboost_pred, atol=1e-7),
            f"d4p and reference SHAP interaction prediction are different (d4p - ref) = {d4p_pred - xgboost_pred}",
        )

    def test_model_predict_shap_contribs_missing_values(self):
        booster = self.xgb_model.get_booster()
        m = d4p.mb.convert_model(booster)
        d4p_pred = m.predict(self.X_nan, pred_contribs=True)
        xgboost_pred = booster.predict(
            xgb.DMatrix(self.X_nan),
            pred_contribs=True,
            approx_contribs=False,
            validate_features=False,
        )
        self.assertTrue(
            np.allclose(d4p_pred, xgboost_pred, atol=1e-7),
            f"d4p and reference SHAP contribution missing value prediction are different (d4p - ref) = {d4p_pred - xgboost_pred}",
        )


class XGBoostClassificationModelBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = make_classification(n_samples=500, n_features=10, random_state=42)
        cls.X_nan = np.array([np.nan] * 20, dtype=np.float32).reshape(2, 10)
        cls.xgb_model = xgb.XGBClassifier(max_depth=5, n_estimators=50, random_state=42)
        cls.xgb_model.fit(cls.X, cls.y)

    def test_model_conversion(self):
        m = d4p.mb.convert_model(self.xgb_model.get_booster())
        self.assertEqual(m.n_classes_, 2)
        self.assertEqual(m.n_features_in_, 10)
        self.assertFalse(m._is_regression)

    def test_model_predict(self):
        m = d4p.mb.convert_model(self.xgb_model.get_booster())
        d4p_pred = m.predict(self.X)
        xgboost_pred = self.xgb_model.predict(self.X)
        self.assertTrue(
            np.allclose(d4p_pred, xgboost_pred, atol=1e-7),
            f"d4p and reference prediction are different (d4p - ref) = {d4p_pred - xgboost_pred}",
        )

    def test_missing_value_support(self):
        m = d4p.mb.convert_model(self.xgb_model.get_booster())
        d4p_pred = m.predict(self.X_nan)
        xgboost_pred = self.xgb_model.predict(self.X_nan)
        self.assertTrue(
            np.allclose(d4p_pred, xgboost_pred, atol=1e-7),
            f"d4p and reference missing value prediction different (d4p - ref) = {d4p_pred - xgboost_pred}",
        )

    def test_model_predict_shap_contribs(self):
        booster = self.xgb_model.get_booster()
        m = d4p.mb.convert_model(booster)
        with self.assertRaises(NotImplementedError):
            m.predict(self.X, pred_contribs=True)

    def test_model_predict_shap_interactions(self):
        booster = self.xgb_model.get_booster()
        m = d4p.mb.convert_model(booster)
        with self.assertRaises(NotImplementedError):
            m.predict(self.X, pred_contribs=True)


if __name__ == "__main__":
    unittest.main()
