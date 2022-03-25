#!/usr/bin/env python
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

try:
    from packaging.version import Version
except ImportError:
    from distutils.version import LooseVersion as Version

from sklearn import __version__ as sklearn_version

if Version(sklearn_version) >= Version("1.0"):
    from ._lof_1_0 import *
else:
    from ._lof_0_22 import *
