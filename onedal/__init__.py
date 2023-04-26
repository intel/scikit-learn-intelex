# ===============================================================================
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
# ===============================================================================

import platform
from daal4py.sklearn._utils import daal_check_version

if "Windows" in platform.system():
    import os
    import sys
    import site
    path_to_env = site.getsitepackages()[0]
    path_to_libs = os.path.join(path_to_env, "Library", "bin")
    if sys.version_info.minor >= 8:
        if 'DALROOT' in os.environ:
            dal_root_redist = os.path.join(
                os.environ['DALROOT'], "redist", "intel64")
            if os.path.exists(dal_root_redist):
                os.add_dll_directory(dal_root_redist)
        os.add_dll_directory(path_to_libs)
    os.environ['PATH'] = path_to_libs + os.pathsep + os.environ['PATH']

try:
    import onedal._onedal_py_dpc as _backend
    _is_dpc_backend = True
except ImportError:
    import onedal._onedal_py_host as _backend
    _is_dpc_backend = False


__all__ = ['decomposition', 'ensemble', 'neighbors', 'primitives', 'svm']

if _is_dpc_backend:
    __all__.append('spmd')

if daal_check_version((2023, 'P', 100)):
    __all__ += ['basic_statistics', 'linear_model']

    if _is_dpc_backend:
        __all__ += ['spmd.basic_statistics', 'spmd.decomposition',
                    'spmd.linear_model', 'spmd.neighbors']
