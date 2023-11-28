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


# simple storage for hyperparameters
hyperparameters_storage = {
    "linear_regression": {"train": {"cpu_macro_block": None, "gpu_macro_block": None}}
}


def set_hyperparameter(algorithm, op, name, value):
    if name not in hyperparameters_storage[algorithm][op].keys():
        raise ValueError(f"Hyperparameter '{name}' doesn't exist in {algorithm}.{op}")
    hyperparameters_storage[algorithm][op][name] = value


def get_hyperparameters(algorithm, op):
    return {
        key: value
        for key, value in hyperparameters_storage[algorithm][op].items()
        if value is not None
    }
