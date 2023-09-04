#!/usr/bin/env python
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

try:
    import optuna.logging
    from optuna.distributions import (
        CategoricalDistribution,
        FloatDistribution,
        IntDistribution,
    )
    from optuna.integration import OptunaSearchCV

    optuna_is_available = True
except (ImportError, ModuleNotFoundError):
    optuna_is_available = False

from warnings import warn


def _translate_param_distributions_to_optuna(input_param_distributions):
    distr_map = {
        "categorical": CategoricalDistribution,
        "int": IntDistribution,
        "float": FloatDistribution,
    }
    output_param_distributions = {}
    for param_name, (distr_type, distr_kwargs) in input_param_distributions.items():
        if distr_type not in distr_map.keys():
            raise ValueError(f"Unknown '{distr_type}' distribution type.")
        output_param_distributions[param_name] = distr_map[distr_type](**distr_kwargs)
    return output_param_distributions


class TuningBase:
    def tune(self, X, y=None, param_distributions=None, **search_params):
        warn(
            "Tuning functionality in sklearnex is experimental and "
            "doesn't guarantee stable API."
        )
        if not hasattr(self, "_get_param_distributions"):
            raise NotImplementedError(
                "Method to receive parameter distributions "
                f"is not implemented in {self.__class__}."
            )
        else:
            param_distributions = search_params.get("param_distributions", None)
            param_distributions = (
                self._get_param_distributions()
                if param_distributions is None
                else param_distributions
            )
            if optuna_is_available:
                param_distributions = _translate_param_distributions_to_optuna(
                    param_distributions
                )
                optuna.logging.set_verbosity(optuna.logging.WARNING)
                search = OptunaSearchCV(self, param_distributions, **search_params)
                search.fit(X, y)
                self = search.best_estimator_
                return search
            else:
                raise NotImplementedError(
                    "Fallback branch for RandomizedSearchCV is not implemented."
                )
