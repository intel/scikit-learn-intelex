#*******************************************************************************
# Copyright 2014-2020 Intel Corporation
# All Rights Reserved.
#
# This software is licensed under the Apache License, Version 2.0 (the
# "License"), the following terms apply:
#
# You may not use this file except in compliance with the License.  You may
# obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# See the License for the specific language governing permissions and
# limitations under the License.
#*******************************************************************************

from sklearn.metrics import * 
import daal4py as d4p
import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

def iris_with_intercept():
    X, y = load_iris(return_X_y=True)
    flag = True
    clf = LogisticRegression(fit_intercept=flag, max_iter=10000, random_state=0).fit(X, y)
    beta = clf.coef_
    #print(beta)
    #print(clf.intercept_)
    builder = d4p.logistic_regression_model_builder(beta.shape[1], beta.shape[0])
    builder.setBeta(clf.coef_, clf.intercept_, flag)

    pred = d4p.logistic_regression_prediction(nClasses=beta.shape[0])
    #print(builder.model)
    predict_result = pred.compute(X, builder.model)
    #print(predict_result.prediction)


def iris_without_intercept():
    X, y = load_iris(return_X_y=True)
    flag = False
    clf = LogisticRegression(fit_intercept=flag, max_iter=10000, random_state=0).fit(X, y)
    beta = clf.coef_
    #print(beta)
    #print(clf.intercept_)
    builder = d4p.logistic_regression_model_builder(beta.shape[1], beta.shape[0])
    builder.setBeta(clf.coef_, clf.intercept_, flag)

    pred = d4p.logistic_regression_prediction(nClasses=beta.shape[0])
    #print(builder.model)
    predict_result = pred.compute(X, builder.model)
    #print(predict_result.prediction)


def breast_cancer_with_intercept():
    X, y = load_breast_cancer(return_X_y=True)
    flag = True
    clf = LogisticRegression(fit_intercept=flag, max_iter=10000, random_state=0).fit(X, y)
    beta = clf.coef_
    #print(beta)
    #print(clf.intercept_)
    builder = d4p.logistic_regression_model_builder(beta.shape[1], beta.shape[0])
    builder.setBeta(clf.coef_, clf.intercept_, flag)

    pred = d4p.logistic_regression_prediction(nClasses=2)
    #print(builder.model)
    predict_result = pred.compute(X, builder.model)
    #print(predict_result.prediction)


def breast_cancer_without_intercept():
    X, y = load_breast_cancer(return_X_y=True)
    flag = False
    clf = LogisticRegression(fit_intercept=flag, max_iter=10000, random_state=0).fit(X, y)
    beta = clf.coef_
    #print(beta)
    #print(clf.intercept_)
    builder = d4p.logistic_regression_model_builder(beta.shape[1], beta.shape[0])
    builder.setBeta(clf.coef_, clf.intercept_, flag)

    pred = d4p.logistic_regression_prediction(nClasses=2)
    #print(builder.model)
    predict_result = pred.compute(X, builder.model)
    #print(predict_result.prediction)


if __name__ == "__main__":
    iris_with_intercept()
    iris_without_intercept()
    breast_cancer_with_intercept()
    breast_cancer_without_intercept()