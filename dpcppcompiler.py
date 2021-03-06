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

import distutils.ccompiler
import sys

_compiler_module, _compiler_class, _ = distutils.ccompiler.compiler_class["msvc"]

_msvc_compiler_module_name = "distutils." + _compiler_module
__import__(_msvc_compiler_module_name)
_msvc_compiler_module = sys.modules[_msvc_compiler_module_name]
cls = vars(_msvc_compiler_module)[_compiler_class]


class DPCPPCompiler(cls):
    def initialize(self):
        super().initialize()
        self.cc = "clang-cl.exe"
        self.cxx = "clang-cl.exe"
        self.linker = "lld-link.exe"
        self.linker_so = "lld-link.exe"
