# ==============================================================================
# Copyright 2021 Intel Corporation
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

import os
import random
import string

import pytest
import sklearn

import sklearnex
from onedal import _is_dpc_backend


def test_get_config_contains_sklearn_params():
    skex_config = sklearnex.get_config()
    sk_config = sklearn.get_config()

    assert all(value in skex_config.keys() for value in sk_config.keys())


def test_set_config_works():
    default_config = sklearnex.get_config()
    sklearnex.set_config(
        assume_finite=True, target_offload="cpu:0", allow_fallback_to_host=True
    )

    config = sklearnex.get_config()
    assert config["target_offload"] == "cpu:0"
    assert config["allow_fallback_to_host"]
    assert config["assume_finite"]
    sklearnex.set_config(**default_config)


gpu_mark = pytest.mark.skipif(
    not _is_dpc_backend, reason="compute_mode is only supported with the dpc backend"
)


@pytest.mark.parametrize(
    "setting",
    [
        pytest.param("standard", marks=gpu_mark),
        pytest.param("FORCE_ALTERNATE", marks=gpu_mark),
        pytest.param(
            ["FLOAT_TO_BF16", "float_to_bf16x2", "float_to_bf16x3"], marks=gpu_mark
        ),
        pytest.param(
            "float_to_bf16,float_to_bf16x2,float_to_bf16x3",
            marks=gpu_mark,
        ),
        pytest.param("any", marks=gpu_mark),
    ],
)
def test_set_compute_mode(setting):
    default_config = sklearnex.get_config()
    sklearnex.set_config(compute_mode=setting)

    config = sklearnex.get_config()
    setting = (
        ",".join(setting) if type(setting) is not str else setting
    )  # convert to internal representation
    assert config["compute_mode"] == setting.lower()
    assert os.environ.get("DAL_BLAS_COMPUTE_MODE", "STANDARD") == setting.upper()
    sklearnex.set_config(**default_config)


# This test has the possibility of an erronous failure albeit vanishingly small
@pytest.mark.skipif(
    not _is_dpc_backend, reason="compute_mode is only supported with the dpc backend"
)
def test_infinite_monkey_compute_mode():
    setting = "".join(random.choices(string.ascii_letters, k=random.randrange(25)))
    default_config = sklearnex.get_config()
    try:
        sklearnex.set_config(compute_mode=setting)
    except ValueError:
        pass

    config = sklearnex.get_config()
    assert config["compute_mode"] == "standard"
    assert os.environ.get("DAL_BLAS_COMPUTE_MODE", None) is None
    sklearnex.set_config(**default_config)
