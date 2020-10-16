#*******************************************************************************
# Copyright 2020 Intel Corporation
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

import daal4py as d4p
import numpy as np
from daal4py.sklearn._utils import daal_check_version
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

def main():
    X, y = load_iris(return_X_y=True)
    n_classes=3
    clf = LogisticRegression(fit_intercept=True, max_iter=1000, random_state=0).fit(X, y)
    builder = d4p.logistic_regression_model_builder(n_classes=n_classes, n_features=X.shape[1])
    builder.set_beta(clf.coef_, clf.intercept_)

    alg_pred = d4p.logistic_regression_prediction(nClasses=n_classes)

    pred_daal = alg_pred.compute(X, builder.model).prediction.flatten()
    pred_sklearn = clf.predict(X)

    assert np.allclose(pred_daal, pred_sklearn)


if __name__ == "__main__":
    if daal_check_version((2021, 'P', 1)):
        main()