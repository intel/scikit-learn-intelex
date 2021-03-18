#!/usr/bin/env python
#===============================================================================
# Copyright 2014-2021 Intel Corporation
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

import platform
if "Windows" in platform.system():
    import os
    import sys
    import site
    path_to_env = site.getsitepackages()[0]
    path_to_libs = os.path.join(path_to_env, "Library", "bin")
    if sys.version_info.minor >= 8:
        os.add_dll_directory(path_to_libs)
    os.environ['PATH'] += os.pathsep + path_to_libs

try:
    from _daal4py import *
    from _daal4py import (
        _get__version__,
        _get__daal_link_version__,
        _get__daal_run_version__,
        __has_dist__)
except ImportError as e:
    s = str(e)
    if 'libfabric' in s:
        raise ImportError(
            s + '\n\nActivating your conda environment or sourcing mpivars.'
            '[c]sh/psxevars.[c]sh may solve the issue.\n')
    raise
