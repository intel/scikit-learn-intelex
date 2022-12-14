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


def build_cpp(cc, cxx, sources, targetprefix, targetname, targetsuffix, libs, libdirs,
              includes, eca, ela, defines, installpath=''):
    import shutil
    import subprocess
    from sysconfig import get_paths as gp
    from os.path import basename

    log.info(f'building cpp target {targetname}...')

    include_dir_plat = ['-I' + incdir for incdir in includes]
    if IS_WIN:
        eca += ['/EHsc']
        lib_prefix = ''
        lib_suffix = '.lib'
        obj_ext = '.obj'
        libdirs += [jp(gp()['data'], 'libs')]
        library_dir_plat = ['/link'] + [f'/LIBPATH:{libdir}' for libdir in libdirs]
        additional_linker_opts = [
            '/DLL',
            f'/OUT:{targetprefix}{targetname}{targetsuffix}'
        ]
    else:
        eca += ['-fPIC']
        ela += ['-shared']
        lib_prefix = '-l'
        lib_suffix = ''
        obj_ext = '.o'
        library_dir_plat = ['-L' + libdir for libdir in libdirs]
        additional_linker_opts = ['-o', f'{targetprefix}{targetname}{targetsuffix}']
    eca += ['-c']
    libs = [f'{lib_prefix}{str(item)}{lib_suffix}' for item in libs]

    d4p_dir = os.getcwd()
    build_dir = os.path.join(d4p_dir, f"build_{targetname}")

    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    os.mkdir(build_dir)
    os.chdir(build_dir)

    objfiles = [basename(f).replace('.cpp', obj_ext) for f in sources]
    for i, cppfile in enumerate(sources):
        if IS_WIN:
            out = [f'/Fo{objfiles[i]}']
        else:
            out = ['-o', objfiles[i]]
        cmd = [cc] + eca + include_dir_plat + [f'{d4p_dir}/{cppfile}'] + out + defines
        log.info(subprocess.list2cmdline(cmd))
        subprocess.check_call(cmd)

    cmd = [cxx] + ela + objfiles + library_dir_plat + libs + additional_linker_opts
    log.info(subprocess.list2cmdline(cmd))
    subprocess.check_call(cmd)
    shutil.copy(f'{targetprefix}{targetname}{targetsuffix}',
                os.path.join(d4p_dir, installpath))
    if IS_WIN:
        target_lib_suffix = targetsuffix.replace('.dll', '.lib')
        shutil.copy(f'{targetprefix}{targetname}{target_lib_suffix}',
                    os.path.join(d4p_dir, installpath))
    os.chdir(d4p_dir)


def custom_build_cmake_clib(iface, cxx=None):
    import pybind11

    root_dir = os.path.normpath(jp(os.path.dirname(__file__), ".."))
    log.info(f"Project directory is: {root_dir}")

    builder_directory = jp(root_dir, "scripts")
    abs_build_temp_path = jp(root_dir, "build", f"backend_{iface}")
    install_directory = jp(root_dir, "onedal")
    log.info(f"Builder directory: {builder_directory}")
    log.info(f"Install directory: {install_directory}")

    cmake_generator = "-GNinja" if IS_WIN else ""
    python_include = get_python_inc()
    win_python_path_lib = os.path.abspath(jp(get_config_var('LIBDEST'), "..", "libs"))
    python_library_dir = win_python_path_lib if IS_WIN else get_config_var('LIBDIR')
    numpy_include = np.get_include()

    if iface == 'dpc':
        if IS_WIN:
            cxx = 'icx'
        else:
            cxx = 'icpx'
    elif cxx is None:
        raise RuntimeError('CXX compiler shall be specified')

    cmake_args = [
        "cmake",
        cmake_generator,
        "-S" + builder_directory,
        "-B" + abs_build_temp_path,
        "-DCMAKE_CXX_COMPILER=" + cxx,
        "-DCMAKE_INSTALL_PREFIX=" + install_directory,
        "-DCMAKE_PREFIX_PATH=" + install_directory,
        "-DIFACE=" + iface,
        "-DPYTHON_INCLUDE_DIR=" + python_include,
        "-DNUMPY_INCLUDE_DIRS=" + numpy_include,
        "-DPYTHON_LIBRARY_DIR=" + python_library_dir,
        "-DoneDAL_INCLUDE_DIRS=" + jp(os.environ['DALROOT'], 'include'),
        "-DoneDAL_LIBRARY_DIR=" + jp(os.environ['DALROOT'], 'lib', 'intel64'),
        "-Dpybind11_DIR=" + pybind11.get_cmake_dir(),
    ]

    if iface == 'dpc':
        cmake_args += ["-DCMAKE_C_COMPILER_WORKS=1",
                       "-DCMAKE_CXX_COMPILER_WORKS=1"]

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
