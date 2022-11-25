#!/usr/bin/env python
#===============================================================================
# Copyright 2020 Intel Corporation
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
    import sysconfig

    current_path = os.path.dirname(__file__)

    sitepackages_path = sysconfig.get_paths()['purelib']
    installed_package_path = os.path.join(sitepackages_path, 'daal4py', 'oneapi')
    if sys.version_info.minor >= 8:
        if 'DPCPPROOT' in os.environ:
            dpcpp_rt_root_bin = os.path.join(os.environ['DPCPPROOT'], "windows", "bin")
            dpcpp_rt_root_redist = os.path.join(os.environ['DPCPPROOT'], "windows",
                                                "redist", "intel64_win", "compiler")
            if os.path.exists(dpcpp_rt_root_bin):
                os.add_dll_directory(dpcpp_rt_root_bin)
            if os.path.exists(dpcpp_rt_root_redist):
                os.add_dll_directory(dpcpp_rt_root_redist)
        os.add_dll_directory(current_path)
        if os.path.exists(installed_package_path):
            os.add_dll_directory(installed_package_path)
    os.environ['PATH'] = current_path + os.pathsep + os.environ['PATH']
    os.environ['PATH'] = installed_package_path + os.pathsep + os.environ['PATH']

try:
    from daal4py._oneapi import *
    from daal4py._oneapi import (
        _get_sycl_ctxt,
        _get_device_name_sycl_ctxt,
        _get_sycl_ctxt_params,
        _get_in_sycl_ctxt
    )
except ModuleNotFoundError:
    raise
except ImportError:
    import daal4py
    version = daal4py._get__version__()[1:-1].split(', ')
    major_version, minor_version = version[0], version[1]
    raise ImportError(
        f'dpcpp_cpp_rt >= {major_version}.{minor_version} '
        'has to be installed or upgraded to use this module.\n'
        'You can download or upgrade it using the following commands:\n'
        f'`pip install --upgrade dpcpp_cpp_rt>={major_version}.{minor_version}.*` '
        'or '
        f'`conda install -c intel dpcpp_cpp_rt>={major_version}.{minor_version}.*`'
    )
