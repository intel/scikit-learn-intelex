# ===============================================================================
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
# ===============================================================================

import warnings
from functools import update_wrapper

from .._config import config_context, get_config


class _FuncWrapper:
    """Load the global configuration before calling the function."""

    def __init__(self, function):
        self.function = function
        update_wrapper(self, self.function)

    def with_config(self, config):
        self.config = config
        return self

    def __call__(self, *args, **kwargs):
        config = getattr(self, "config", None)
        if config is None:
            warnings.warn(
                "`sklearn.utils.parallel.delayed` should be used with "
                "`sklearn.utils.parallel.Parallel` to make it possible to propagate "
                "the scikit-learn configuration of the current thread to the "
                "joblib workers.",
                UserWarning,
            )
            config = {}
        with config_context(**config):
            return self.function(*args, **kwargs)


class _FuncWrapperOld:
    """Load the global configuration before calling the function."""

    def __init__(self, function):
        self.function = function
        self.config = get_config()
        update_wrapper(self, self.function)

    def __call__(self, *args, **kwargs):
        with config_context(**self.config):
            return self.function(*args, **kwargs)
