.. ******************************************************************************
.. * Copyright 2020 Intel Corporation
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
Model Builders for Gradinet Boosting Framewotrks
###############################################
Introduction
---------
Gradient boosting on decision trees is one of the most accurate and efficient 
machine learning algorithms for classification and regression. There are many 
implementations of gradient boosting, but the most popular are the XGBoost, 
LightGBM and CatBoost frameworks.
daal4py Model Builders deliver accelerated XGBoost, LightGBM and CatBoost 
models inference. Inference performed by oneDAL GBT implementation tuned 
for best performance on Intel Architecture. 

Conversion
---------
First step of process is conversion of already trained model. There is simmilar 
API for different frameworks
XGBoost::

  import daal4py as d4p
  d4p_model = d4p.get_gbt_model_from_xgboost(xgb_model)

LightGBM::

  import daal4py as d4p
  d4p_model = d4p.get_gbt_model_from_lightgbm(lgb_model)

CatBoost::

  import daal4py as d4p
  d4p_model = d4p.get_gbt_model_from_catboost(cb_model)

It's requered to convert model only once and then it will be used for inference

Classification and Regression inference
---------
GBT implementaiton in daal4py assumes separate APIs for classification and regression.
Corresponding API should be explicitly specified and should match corresponding problem 
in initial framework.

Classification::

    d4p_cls_algo = d4p.gbt_classification_prediction(
        nClasses=params['classes_count'],
        resultsToEvaluate="computeClassLabels",
        fptype='float'
    )

Regression::
    d4p_reg_algo = d4p.gbt_regression_prediction()

As a next step d4py algorithm object need compute method to be called. 
Both data and previously converted model should be passed with results of predection 
avaialbe within .prediction parameter.

Compute::

    d4p_predictions = d4p_reg_algo.compute(X_test, d4p_model).prediction

As alternative here is one line variant of same code::
    d4p_prediction = d4p.gbt_regression_prediction().compute(X_test, d4p_model).prediction


Limitations
---------------------------------
Missing Values (NaN)
Note that there is temporary limitation on the use of missing values 
(NaN) during training and prediction. This problem already adressed in 
master and would be avaialbe as part of 2023.2 release.

Examples
---------------------------------
Model Builders models conversion

- `XGBoost model conversion <https://github.com/intel/scikit-learn-intelex/blob/master/examples/daal4py/gbt_cls_model_create_from_xgboost_batch.py>`_
- `LightGBM model conversion <https://github.com/intel/scikit-learn-intelex/blob/master/examples/daal4py/gbt_cls_model_create_from_lightgbm_batch.py>`_
- `CatBoost model conversion <https://github.com/intel/scikit-learn-intelex/blob/master/examples/daal4py/gbt_cls_model_create_from_catboost_batch.py>`_

Articles and blog posts
---------------------------------

-  `Improving the Performance of XGBoost and LightGBM Inference <https://medium.com/intel-analytics-software/improving-the-performance-of-xgboost-and-lightgbm-inference-3b542c03447e>` _
