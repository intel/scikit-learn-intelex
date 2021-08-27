#===============================================================================
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
#===============================================================================

import sklearn
import sklearnex


def test_get_config_contains_sklearn_params():
    skex_config = sklearnex.get_config()
    sk_config = sklearn.get_config()

    assert all(value in skex_config.keys() for value in sk_config.keys())


def test_set_config_works():
    default_config = sklearnex.get_config()
    sklearnex.set_config(assume_finite=True,
                         target_offload='cpu:0',
                         allow_fallback_to_host=True)

    config = sklearnex.get_config()
    assert config['target_offload'] == 'cpu:0'
    assert config['allow_fallback_to_host']
    assert config['assume_finite']
    sklearnex.set_config(**default_config)
