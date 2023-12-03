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

from daal4py.sklearn._utils import daal_check_version
from onedal import _backend


class HyperParameters:
    def __init__(self, algorithm, op, setters, getters, backend):
        self.algorithm = algorithm
        self.op = op
        self.setters = setters
        self.getters = getters
        self.backend = backend

    def __getattr__(self, __name):
        if __name in ["algorithm", "op", "setters", "getters", "backend"]:
            return super().__getattr__(__name)
        elif __name in self.getters.keys():
            return self.getters[__name]()
        else:
            raise ValueError(
                f"Unknown '{__name}' name in "
                f"'{self.algorithm}.{self.op}' hyperparameters"
            )

    def __setattr__(self, __name, __value):
        if __name in ["algorithm", "op", "setters", "getters", "backend"]:
            super().__setattr__(__name, __value)
        elif __name in self.setters.keys():
            self.setters[__name](__value)
        else:
            raise ValueError(
                f"Unknown '{__name}' name in "
                f"'{self.algorithm}.{self.op}' hyperparameters"
            )


def get_methods_with_prefix(obj, prefix):
    return {
        method.replace(prefix, ""): getattr(obj, method)
        for method in filter(lambda f: f.startswith(prefix), dir(obj))
    }


hyperparameters_backend = {
    (
        "linear_regression",
        "train",
    ): _backend.linear_model.regression.train_hyperparameters(),
    ("covariance", "compute"): _backend.covariance.compute_hyperparameters(),
}
hyperparameters_map = {}

for (algorithm, op), hyperparameters in hyperparameters_backend.items():
    setters = get_methods_with_prefix(hyperparameters, "set_")
    getters = get_methods_with_prefix(hyperparameters, "get_")

    if set(setters.keys()) != set(getters.keys()):
        raise ValueError(
            f"Setters and getters in '{algorithm}.{op}' "
            "hyperparameters wrapper do not correspond."
        )

    hyperparameters_map[(algorithm, op)] = HyperParameters(
        algorithm, op, setters, getters, hyperparameters
    )


def get_hyperparameters(algorithm, op):
    return hyperparameters_map[(algorithm, op)]
