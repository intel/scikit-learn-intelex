#===============================================================================
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
#===============================================================================

# daal4py Model builders API

import daal4py as d4p


class GBTDAALModel(d4p.GBTDAALBaseModel):
    def predict(self, X, fptype="float"):
        if self._is_regression:
            return self._predict_regression(X, fptype)
        else:
            return self._predict_classification(X, fptype, "computeClassLabels")

    def predict_proba(self, X, fptype="float"):
        if self._is_regression:
            raise NotImplementedError("Can't predict probabilities for regression task")
        else:
            return self._predict_classification(X, fptype, "computeClassProbabilities")


def convert_model(model):
    gbm = GBTDAALModel()
    gbm._convert_model(model)

    gbm._is_regression = isinstance(gbm.daal_model_, d4p.gbt_regression_model)

    return gbm
