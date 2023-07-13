.. ******************************************************************************
.. * Copyright 2023 Intel Corporation
.. *
.. * Licensed under the Apache License, Version 2.0 (the "License");
.. * you may not use this file except in compliance with the License.
.. * You may obtain a copy of the License at
.. *
.. *     http://www.apache.org/licenses/LICENSE-2.0
.. *
.. * Unless required by applicable law or agreed to in writing, software
.. * distributed under the License is distributed on an "AS IS" BASIS,
.. * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. * See the License for the specific language governing permissions and
.. * limitations under the License.
.. *******************************************************************************/

.. _model-builders:

###############################################
Model Builders for the Gradient Boosting Frameworks
###############################################
Introduction
------------------
Gradient boosting on decision trees is one of the most accurate and efficient 
machine learning algorithms for classification and regression. 
The most popular implementations of it are the XGBoost*, 
LightGBM*, and CatBoost* frameworks.
daal4py Model Builders deliver the accelerated
models inference of those frameworks. The inference is performed by the oneDAL GBT implementation tuned 
for the best performance on the Intel(R) Architecture. 

Conversion
---------
The first step is to convert already trained model. There are similar 
APIs for different frameworks. 
XGBoost::

  import daal4py as d4p
  d4p_model = d4p.get_gbt_model_from_xgboost(xgb_model)

LightGBM::

  import daal4py as d4p
  d4p_model = d4p.get_gbt_model_from_lightgbm(lgb_model)

CatBoost::

  import daal4py as d4p
  d4p_model = d4p.get_gbt_model_from_catboost(cb_model)

.. note:: Convert model only once and then use it for the inference.

Classification and Regression Inference
---------
GBT implementation in daal4py assumes separate APIs for the classification and regression.
Specify the corresponding API and match the corresponding problem 
in the initial framework.

Classification::

    d4p_cls_algo = d4p.gbt_classification_prediction(
        nClasses=params['classes_count'],
        resultsToEvaluate="computeClassLabels",
        fptype='float'
    )

Regression::
    d4p_reg_algo = d4p.gbt_regression_prediction()

Next, daal4py algorithm object needs compute method to be called. 
Both the data and the previously converted model should be passed with the results of the prediction 
available within the  ``.prediction`` parameter.

Compute::

    d4p_predictions = d4p_reg_algo.compute(X_test, d4p_model).prediction

The one-line variant of the same code::
    d4p_prediction = d4p.gbt_regression_prediction().compute(X_test, d4p_model).prediction


Limitations
---------------------------------
Missing Values (NaN)
Note that there is temporary limitation on the use of missing values 
(NaN) during training and prediction. This problem is addressed on 
the master branch and to be available in the 2023.2 release.

Examples
---------------------------------
Model Builders models conversion

- `XGBoost model conversion <https://github.com/intel/scikit-learn-intelex/blob/master/examples/daal4py/model_builders_xgboost.py>`_
- `LightGBM model conversion <https://github.com/intel/scikit-learn-intelex/blob/master/examples/daal4py/model_builders_lightgbm.py>`_
- `CatBoost model conversion <https://github.com/intel/scikit-learn-intelex/blob/master/examples/daal4py/model_builders_catboost.py>`_

Articles and Blog Posts
---------------------------------

-  `Improving the Performance of XGBoost and LightGBM Inference <https://medium.com/intel-analytics-software/improving-the-performance-of-xgboost-and-lightgbm-inference-3b542c03447e>` _
