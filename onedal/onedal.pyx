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

# distutils: language = c++
#cython: language_level=2

# Import the Python-level symbols of numpy

# Import the C-level symbols of numpy
cimport numpy as npc

npc.import_array()

include "svm/svm.pyx"
include "prims/kernel_functions.pyx"
include "common/policy.pyx"
