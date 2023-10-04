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
import shap
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
        X, y = make_regression(n_samples=2, n_features=10, random_state=42)
        cls.X_test = X[:2, :]
        cls.X_nan = np.array([np.nan] * 20, dtype=np.float32).reshape(2, 10)
        cls.xgb_model = xgb.XGBRegressor(max_depth=5, n_estimators=50, random_state=42)
        cls.xgb_model.fit(X, y)

    def test_model_conversion(self):
        m = d4p.mb.convert_model(self.xgb_model.get_booster())
        # XGBoost treats regression as 0 classes, LightGBM 1 class
        # For us, it does not make a difference and both are acceptable
        self.assertEqual(m.n_classes_, 0)
        self.assertEqual(m.n_features_in_, 10)
        self.assertTrue(m._is_regression)

    def test_model_predict(self):
        m = d4p.mb.convert_model(self.xgb_model.get_booster())
        d4p_pred = m.predict(self.X_test)
        xgboost_pred = self.xgb_model.predict(self.X_test)
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
        d4p_pred = m.predict(self.X_test, pred_contribs=True)
        xgboost_pred = booster.predict(
            xgb.DMatrix(self.X_test),
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
        d4p_pred = m.predict(self.X_test, pred_interactions=True)
        xgboost_pred = booster.predict(
            xgb.DMatrix(self.X_test),
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
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
        cls.X_test = X[:2, :]
        cls.X_nan = np.array([np.nan] * 20, dtype=np.float32).reshape(2, 10)
        cls.xgb_model = xgb.XGBClassifier(max_depth=5, n_estimators=50, random_state=42)
        cls.xgb_model.fit(X, y)

    def test_model_conversion(self):
        m = d4p.mb.convert_model(self.xgb_model.get_booster())
        self.assertEqual(m.n_classes_, 2)
        self.assertEqual(m.n_features_in_, 10)
        self.assertFalse(m._is_regression)

    def test_model_predict(self):
        m = d4p.mb.convert_model(self.xgb_model.get_booster())
        d4p_pred = m.predict(self.X_test)
        xgboost_pred = self.xgb_model.predict(self.X_test)
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
            m.predict(self.X_test, pred_contribs=True)

    def test_model_predict_shap_interactions(self):
        booster = self.xgb_model.get_booster()
        m = d4p.mb.convert_model(booster)
        with self.assertRaises(NotImplementedError):
            m.predict(self.X_test, pred_contribs=True)


class LightGBMRegressionModelBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)
        cls.X_test = X[:2, :]
        cls.X_nan = np.array([np.nan] * 20, dtype=np.float32).reshape(2, 10)
        params = {
            "task": "train",
            "boosting": "gbdt",
            "objective": "regression",
            "num_leaves": 10,
            "learning_rage": 0.05,
            "metric": {"l2", "l1"},
            "verbose": -1,
        }
        cls.lgbm_model = lgbm.train(params, train_set=lgbm.Dataset(X, y))

    def test_model_conversion(self):
        m = d4p.mb.convert_model(self.lgbm_model)
        # XGBoost treats regression as 0 classes, LightGBM 1 class
        # For us, it does not make a difference and both are acceptable
        self.assertEqual(m.n_classes_, 1)
        self.assertEqual(m.n_features_in_, 10)
        self.assertTrue(m._is_regression)

    def test_model_predict(self):
        m = d4p.mb.convert_model(self.lgbm_model)
        d4p_pred = m.predict(self.X_test)
        lgbm_pred = self.lgbm_model.predict(self.X_test)
        max_diff = np.absolute(d4p_pred - lgbm_pred).reshape(1, -1).max()
        self.assertLess(max_diff, 1e-7)

    def test_missing_value_support(self):
        m = d4p.mb.convert_model(self.lgbm_model)
        d4p_pred = m.predict(self.X_nan)
        lgbm_pred = self.lgbm_model.predict(self.X_nan)
        max_diff = np.absolute(d4p_pred - lgbm_pred).reshape(1, -1).max()
        self.assertLess(max_diff, 1e-7)

    def test_model_predict_shap_contribs(self):
        m = d4p.mb.convert_model(self.lgbm_model)
        d4p_pred = m.predict(self.X_test, pred_contribs=True)
        lgbm_pred = self.lgbm_model.predict(self.X_test, pred_contrib=True)
        self.assertTrue(
            d4p_pred.shape == lgbm_pred.shape,
            f"d4p and reference SHAP contribution shape is different {d4p_pred.shape} != {lgbm_pred.shape}",
        )
        max_diff = np.absolute(d4p_pred - lgbm_pred).reshape(1, -1).max()
        self.assertLess(max_diff, 1e-7)

    def test_model_predict_shap_interactions(self):
        m = d4p.mb.convert_model(self.lgbm_model)
        # SHAP Python package drops bias terms from the returned matrix, therefore we drop the final row & column
        d4p_pred = m.predict(self.X_test, pred_interactions=True)[:, :-1, :-1]
        explainer = shap.TreeExplainer(self.lgbm_model)
        shap_pred = explainer.shap_interaction_values(self.X_test)
        self.assertTrue(
            d4p_pred.shape == shap_pred.shape,
            f"d4p and reference SHAP contribution shape is different {d4p_pred.shape} != {shap_pred.shape}",
        )
        max_diff = np.absolute(d4p_pred - shap_pred).reshape(1, -1).max()
        self.assertLess(max_diff, 1e-7)

    def test_model_predict_shap_contribs_missing_values(self):
        m = d4p.mb.convert_model(self.lgbm_model)
        d4p_pred = m.predict(self.X_nan, pred_contribs=True)
        lgbm_pred = self.lgbm_model.predict(self.X_nan, pred_contrib=True)
        self.assertTrue(
            d4p_pred.shape == lgbm_pred.shape,
            f"d4p and reference SHAP contribution shape is different {d4p_pred.shape} != {lgbm_pred.shape}",
        )
        max_diff = np.absolute(d4p_pred - lgbm_pred).reshape(1, -1).max()
        self.assertLess(max_diff, 1e-7)


class LightGBMClassificationModelBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = make_classification(
            random_state=3, n_classes=3, n_informative=3, n_features=10
        )
        cls.X_test = X[:2, :]
        cls.X_nan = np.array([np.nan] * 20, dtype=np.float32).reshape(2, 10)
        params = {
            "n_estimators": 10,
            "task": "train",
            "boosting": "gbdt",
            "objective": "multiclass",
            "num_leaves": 4,
            "num_class": 3,
            "verbose": -1,
        }
        cls.lgbm_model = lgbm.train(params, train_set=lgbm.Dataset(X, y))

    def test_model_conversion(self):
        m = d4p.mb.convert_model(self.lgbm_model)
        self.assertEqual(m.n_classes_, 3)
        self.assertEqual(m.n_features_in_, 10)
        self.assertTrue(m._is_regression)

    def test_model_predict(self):
        m = d4p.mb.convert_model(self.lgbm_model)
        d4p_pred = m.predict(self.X_test)
        lgbm_pred = np.argmax(self.lgbm_model.predict(self.X_test), axis=1)
        self.assertTrue((d4p_pred == lgbm_pred).all())

    def test_missing_value_support(self):
        m = d4p.mb.convert_model(self.lgbm_model)
        d4p_pred = m.predict(self.X_nan)
        lgbm_pred = np.argmax(self.lgbm_model.predict(self.X_test), axis=1)
        self.assertTrue((d4p_pred == lgbm_pred).all())

    def test_model_predict_shap_contribs(self):
        m = d4p.mb.convert_model(self.lgbm_model)
        with self.assertRaises(NotImplementedError):
            m.predict(self.X_test, pred_contribs=True)

    def test_model_predict_shap_interactions(self):
        m = d4p.mb.convert_model(self.lgbm_model)
        with self.assertRaises(NotImplementedError):
            m.predict(self.X_test, pred_interactions == True)

    def test_model_predict_shap_contribs_missing_values(self):
        m = d4p.mb.convert_model(self.lgbm_model)
        with self.assertRaises(NotImplementedError):
            m.predict(self.X_nan, pred_contribs=True)


class CatBoostRegressionModelBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)
        cls.X_test = X[:2, :]
        cls.X_nan = np.array([np.nan] * 20, dtype=np.float32).reshape(2, 10)
        params = {
            "reg_lambda": 1,
            "max_depth": 3,
            "num_leaves": 2**3,
            "verbose": 0,
            "objective": "RMSE",
            "learning_rate": 0.3,
            "n_estimators": 25,
        }
        cls.cb_model = cb.CatBoost(params)
        cls.cb_model.fit(X, y, verbose=0)

    def test_model_conversion(self):
        m = d4p.mb.convert_model(self.cb_model)
        self.assertTrue(hasattr(m, "daal_model_"))
        self.assertIsInstance(m.daal_model_, d4p._daal4py.gbt_regression_model)
        self.assertEqual(m.daal_model_.NumberOfFeatures, 10)
        self.assertEqual(m.daal_model_.NumberOfTrees, 25)
        self.assertEqual(m.n_features_in_, 10)
        self.assertTrue(m._is_regression)

    def test_model_predict(self):
        m = d4p.mb.convert_model(self.cb_model)
        d4p_pred = m.predict(self.X_test)
        lgbm_pred = self.cb_model.predict(self.X_test)
        max_diff = np.absolute(d4p_pred - lgbm_pred).reshape(1, -1).max()
        self.assertLess(max_diff, 1e-7)

    def test_missing_value_support(self):
        m = d4p.mb.convert_model(self.cb_model)
        d4p_pred = m.predict(self.X_nan)
        lgbm_pred = self.cb_model.predict(self.X_nan)
        max_diff = np.absolute(d4p_pred - lgbm_pred).reshape(1, -1).max()
        self.assertLess(max_diff, 1e-7)

    def test_model_predict_shap_contribs(self):
        # SHAP value support from CatBoost models is to be added
        with self.assertWarnsRegex(
            Warning,
            "Models converted from CatBoost cannot be used for SHAP value calculation",
        ):
            d4p.mb.convert_model(self.cb_model)


class CatBoostClassificationModelBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = make_classification(
            n_classes=3, n_features=10, n_informative=3, random_state=42
        )
        cls.X_test = X[:2, :]
        cls.X_nan = np.array([np.nan] * 20, dtype=np.float32).reshape(2, 10)
        params = {
            "reg_lambda": 1,
            "max_depth": 3,
            "num_leaves": 2**3,
            "verbose": 0,
            "objective": "MultiClass",
            "learning_rate": 0.3,
            "n_estimators": 25,
        }
        cls.cb_model = cb.CatBoost(params)
        cls.cb_model.fit(X, y, verbose=0)

    def test_model_conversion(self):
        m = d4p.mb.convert_model(self.cb_model)
        self.assertTrue(hasattr(m, "daal_model_"))
        self.assertIsInstance(m.daal_model_, d4p._daal4py.gbt_classification_model)
        self.assertEqual(m.daal_model_.NumberOfFeatures, 10)
        self.assertEqual(m.daal_model_.NumberOfTrees, 3 * 25)
        self.assertEqual(m.n_features_in_, 10)
        self.assertFalse(m._is_regression)

    def test_model_predict(self):
        m = d4p.mb.convert_model(self.cb_model)
        d4p_pred = m.predict(self.X_test)
        cb_pred = self.cb_model.predict(self.X_test, prediction_type="Class").T[0]
        self.assertTrue((d4p_pred == cb_pred).all())

    def test_missing_value_support(self):
        m = d4p.mb.convert_model(self.cb_model)
        d4p_pred = m.predict(self.X_nan)
        cb_pred = self.cb_model.predict(self.X_nan, prediction_type="Class").T[0]
        self.assertTrue((d4p_pred == cb_pred).all())

    def test_model_predict_shap_contribs(self):
        # SHAP value support from CatBoost models is to be added
        with self.assertWarnsRegex(
            Warning,
            "Models converted from CatBoost cannot be used for SHAP value calculation",
        ):
            d4p.mb.convert_model(self.cb_model)


if __name__ == "__main__":
    unittest.main()
