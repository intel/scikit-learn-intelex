#! /usr/bin/env python
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

import os
from os.path import join as jp
import sys
import numpy as np
import subprocess
from distutils import log
from distutils.sysconfig import get_python_inc, get_config_var

IS_WIN = False
IS_MAC = False
IS_LIN = False

if 'linux' in sys.platform:
    IS_LIN = True
elif sys.platform == 'darwin':
    IS_MAC = True
elif sys.platform in ['win32', 'cygwin']:
    IS_WIN = True


def custom_build_cmake_clib(iface):
    import pybind11

    root_dir = os.path.normpath(jp(os.path.dirname(__file__), ".."))
    log.info(f"Project directory is: {root_dir}")

    builder_directory = jp(root_dir, "scripts")
    abs_build_temp_path = jp(root_dir, "build", "backend")
    install_directory = jp(root_dir, "onedal")
    log.info(f"Builder directory: {builder_directory}")
    log.info(f"Install directory: {install_directory}")

    cmake_generator = "-GNinja" if IS_WIN else ""
    python_include = get_python_inc()
    win_python_path_lib = os.path.abspath(jp(get_config_var('LIBDEST'), "..", "libs"))
    python_library_dir = win_python_path_lib if IS_WIN else get_config_var('LIBDIR')
    numpy_include = np.get_include()

    cmake_args = [
        "cmake",
        cmake_generator,
        "-S" + builder_directory,
        "-B" + abs_build_temp_path,
        "-DCMAKE_INSTALL_PREFIX=" + install_directory,
        "-DCMAKE_PREFIX_PATH=" + install_directory,
        "-DIFACE=" + iface,
        "-DPYTHON_INCLUDE_DIR=" + python_include,
        "-DNUMPY_INCLUDE_DIRS=" + numpy_include,
        "-DPYTHON_LIBRARY_DIR=" + python_library_dir,
        "-Dpybind11_DIR=" + pybind11.get_cmake_dir(),
    ]

    import multiprocessing
    cpu_count = multiprocessing.cpu_count()

    make_args = [
        "cmake",
        "--build",
        abs_build_temp_path,
        "-j " + str(cpu_count)
    ]

    make_install_args = [
        "cmake",
        "--install",
        abs_build_temp_path,
    ]

    subprocess.check_call(cmake_args)
    subprocess.check_call(make_args)
    subprocess.check_call(make_install_args)
