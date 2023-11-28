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

import logging
from warnings import warn

from daal4py.sklearn._utils import daal_check_version

# simple storage for hyperparameters
hyperparameters_storage = {
    "linear_regression": {"train": {"cpu_macro_block": 8192, "gpu_macro_block": 16384}}
}


def set_hyperparameter(algorithm, op, name, value):
    if not daal_check_version((2024, "P", 0)):
        warn(f"Hyperparameters are supported starting from 2024.0.0 oneDAL version.")
    if name not in hyperparameters_storage[algorithm][op].keys():
        raise ValueError(f"Hyperparameter '{name}' doesn't exist in {algorithm}.{op}")
    hyperparameters_storage[algorithm][op][name] = value


def get_hyperparameters(algorithm, op):
    res = {
        key: value
        for key, value in hyperparameters_storage[algorithm][op].items()
        if value is not None
    }
    if len(res) > 0:
        logging.getLogger("sklearnex").debug(
            f"Using next hyperparameters for '{algorithm}.{op}': {res}"
        )
    return res
