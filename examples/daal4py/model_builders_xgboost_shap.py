# ==============================================================================
# Copyright 2023 Intel Corporation
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

# daal4py Gradient Boosting Classification model creation and SHAP value
# prediction example

import numpy as np
import xgboost as xgb
from scipy.stats import chisquare
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

import daal4py as d4p


def main():
    # create data
    X, y = make_regression(n_samples=10000, n_features=10, random_state=42)
    X_train, X_test, y_train, _ = train_test_split(X, y, random_state=42)

    # train the model
    xgb_model = xgb.XGBRegressor(
        max_depth=6, n_estimators=100, random_state=42, base_score=0.5
    )
    xgb_model.fit(X_train, y_train)

    # Conversion to daal4py
    daal_model = d4p.mb.convert_model(xgb_model.get_booster())

    # SHAP contributions
    daal_contribs = daal_model.predict(X_test, pred_contribs=True)

    # SHAP interactions
    daal_interactions = daal_model.predict(X_test, pred_interactions=True)

    # XGBoost reference values
    xgb_contribs = xgb_model.get_booster().predict(
        xgb.DMatrix(X_test), pred_contribs=True, validate_features=False
    )
    xgb_interactions = xgb_model.get_booster().predict(
        xgb.DMatrix(X_test), pred_interactions=True, validate_features=False
    )

    return (
        daal_contribs,
        daal_interactions,
        xgb_contribs,
        xgb_interactions,
    )


if __name__ == "__main__":
    daal_contribs, daal_interactions, xgb_contribs, xgb_interactions = main()
    print(f"XGBoost SHAP contributions shape: {xgb_contribs.shape}")
    print(f"daal4py SHAP contributions shape: {daal_contribs.shape}")

    print(f"XGBoost SHAP interactions shape: {xgb_interactions.shape}")
    print(f"daal4py SHAP interactions shape: {daal_interactions.shape}")

    contribution_rmse = np.sqrt(
        np.mean((daal_contribs.reshape(-1, 1) - xgb_contribs.reshape(-1, 1)) ** 2)
    )
    print(f"SHAP contributions RMSE: {contribution_rmse:.2e}")

    interaction_rmse = np.sqrt(
        np.mean((daal_interactions.reshape(-1, 1) - xgb_interactions.reshape(-1, 1)) ** 2)
    )
    print(f"SHAP interactions RMSE: {interaction_rmse:.2e}")
