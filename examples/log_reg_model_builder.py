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

import daal4py as d4p
import numpy as np
from daal4py.sklearn._utils import daal_check_version
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


def main():
    X, y = load_iris(return_X_y=True)
    n_classes = 3

    # set parameters and train
    clf = LogisticRegression(fit_intercept=True, max_iter=1000, random_state=0).fit(X, y)

    #set parameters and call model builder
    builder = d4p.logistic_regression_model_builder(n_classes=n_classes,
                                                    n_features=X.shape[1])
    builder.set_beta(clf.coef_, clf.intercept_)

    # set parameters and compute predictions
    predict_alg = d4p.logistic_regression_prediction(
        nClasses=n_classes,
        resultsToEvaluate="computeClassLabels"
    )
    # set parameters and compute predictions
    predict_result_daal = predict_alg.compute(X, builder.model)
    predict_result_sklearn = clf.predict(X)
    assert np.allclose(predict_result_daal.prediction.flatten(), predict_result_sklearn)
    return (builder, predict_result_daal)


if __name__ == "__main__":
    if daal_check_version(((2021, 'P', 1))):
        (builder, predict_result_daal) = main()
        print("\nLogistic Regression coefficients:\n", builder.model)
        print(
            "\nLogistic regression prediction results (first 10 rows):\n",
            predict_result_daal.prediction[0:10]
        )
        print(
            "\nLogistic regression prediction probabilities (first 10 rows):\n",
            predict_result_daal.probabilities[0:10]
        )
        print(
            "\nLogistic regression prediction log probabilities (first 10 rows):\n",
            predict_result_daal.logProbabilities[0:10]
        )
        print('All looks good!')
