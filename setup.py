#! /usr/bin/env python
#*******************************************************************************
# Copyright 2014-2019 Intel Corporation
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
#******************************************************************************/

# System imports
import os
import subprocess
import sys
import time
from distutils.core import *
from distutils      import sysconfig
from setuptools     import setup, Extension
from os.path import join as jp
from distutils.sysconfig import get_config_vars
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import glob

import numpy as np

d4p_version = os.environ['DAAL4PY_VERSION'] if 'DAAL4PY_VERSION' in os.environ else time.strftime('0.2019.%Y%m%d.%H%M%S')

trues = ['true', 'True', 'TRUE', '1', 't', 'T', 'y', 'Y', 'Yes', 'yes', 'YES']
no_dist = True if 'NO_DIST' in os.environ and os.environ['NO_DIST'] in trues else False
if not no_dist and sys.version_info <= (3, 6):
    print('distributed mode not supported for python version < 3.6\n')
    no_dist = True
no_stream = True if 'NO_STREAM' in os.environ and os.environ['NO_STREAM'] in trues else False
daal_root = os.environ['DAALROOT']
mpi_root = None if no_dist else os.environ['MPIROOT']
dpcpp = True if 'DPCPPROOT' in os.environ else False
dpcpp_root = None if not dpcpp else os.environ['DPCPPROOT']

#itac_root = os.environ['VT_ROOT']
IS_WIN = False
IS_MAC = False
IS_LIN = False

if 'linux' in sys.platform:
    IS_LIN = True
    lib_dir = jp(daal_root, 'lib', 'intel64')
elif sys.platform == 'darwin':
    IS_MAC = True
    lib_dir = jp(daal_root, 'lib')
elif sys.platform in ['win32', 'cygwin']:
    IS_WIN = True
    lib_dir = jp(daal_root, 'Library', 'lib', 'intel64_win')
else:
    assert False, sys.platform + ' not supported'

daal_lib_dir = lib_dir if (IS_MAC or os.path.isdir(lib_dir)) else os.path.dirname(lib_dir)
DAAL_LIBDIRS = [daal_lib_dir]

if no_stream :
    print('\nDisabling support for streaming mode\n')
if no_dist :
    print('\nDisabling support for distributed mode\n')
    DIST_CFLAGS  = []
    DIST_CPPS    = []
    MPI_INCDIRS = []
    MPI_LIBDIRS = []
    MPI_LIBS    = []
    MPI_CPPS     = []
else:
    DIST_CFLAGS  = ['-D_DIST_',]
    DIST_CPPS    = ['src/transceiver.cpp']
    MPI_INCDIRS = [jp(mpi_root, 'include')]
    MPI_LIBDIRS = [jp(mpi_root, 'lib')]
    MPI_LIBNAME = getattr(os.environ, 'MPI_LIBNAME', None)
    if MPI_LIBNAME:
        MPI_LIBS = [MPI_LIBNAME]
    elif IS_WIN:
        if os.path.isfile(jp(mpi_root, 'lib', 'mpi.lib')):
            MPI_LIBS    = ['mpi']
        if os.path.isfile(jp(mpi_root, 'lib', 'impi.lib')):
            MPI_LIBS    = ['impi']
        assert MPI_LIBS, "Couldn't find MPI library"
    else:
        MPI_LIBS    = ['mpi']
    MPI_CPPS = ['src/mpi/mpi_transceiver.cpp']

#Level Zero workaround for oneDAL Beta06
from generator.parse import parse_version

header_path = os.path.join(daal_root, 'include', 'services', 'library_version_info.h')

with open(header_path) as header:
    v = parse_version(header)
    dal_build_version = (int(v[0]), int(v[2]))

if dpcpp:
    DPCPP_CFLAGS = ['-D_DPCPP_']
    DPCPP_LIBS = ['OpenCL', 'sycl', 'daal_sycl']
    if IS_LIN:
        DPCPP_LIBDIRS = [jp(dpcpp_root, 'linux', 'lib')]
    elif IS_WIN:
        DPCPP_LIBDIRS = [jp(dpcpp_root, 'windows', 'lib')]
    if dal_build_version == (2021,6):
        DPCPP_LIBS.append('ze_loader')
        DAAL_LIBDIRS.append('/usr/local/lib')
else:
    DPCPP_CFLAGS = []
    DPCPP_LIBS = []
    DPCPP_LIBDIRS = []

DAAL_DEFAULT_TYPE = 'double'

def get_sdl_cflags():
    if IS_LIN or IS_MAC:
        return DIST_CFLAGS + DPCPP_CFLAGS + ['-fstack-protector', '-fPIC',
                                             '-D_FORTIFY_SOURCE=2', '-Wformat', '-Wformat-security',]
    elif IS_WIN:
        return DIST_CFLAGS + DPCPP_CFLAGS + ['-GS',]

def get_sdl_ldflags():
    if IS_LIN:
        return ['-Wl,-z,noexecstack,-z,relro,-z,now',]
    elif IS_MAC:
        return []
    elif IS_WIN:
        return ['-NXCompat', '-DynamicBase']

def get_type_defines():
    daal_type_defines = ['DAAL_ALGORITHM_FP_TYPE', 'DAAL_SUMMARY_STATISTICS_TYPE', 'DAAL_DATA_TYPE']
    return ["-D{}={}".format(d, DAAL_DEFAULT_TYPE) for d in daal_type_defines]

def getpyexts():
    include_dir_plat = [os.path.abspath('./src'), daal_root + '/include',]
    # FIXME it is a wrong place for this dependency
    if not no_dist:
        include_dir_plat.append(mpi_root + '/include')
    using_intel = os.environ.get('cc', '') in ['icc', 'icpc', 'icl']
    eca = ['-DPY_ARRAY_UNIQUE_SYMBOL=daal4py_array_API', '-DD4P_VERSION="'+d4p_version+'"', '-DNPY_ALLOW_THREADS=1'] + get_type_defines()
    ela = []

    if using_intel and IS_WIN:
        include_dir_plat.append(jp(os.environ.get('ICPP_COMPILER16', ''), 'compiler', 'include'))
        eca += ['-std=c++11', '-w', '/MD']
    elif not using_intel and IS_WIN:
        eca += ['-wd4267', '-wd4244', '-wd4101', '-wd4996', '/MD']
    else:
        eca += ['-std=c++11', '-w',]  # '-D_GLIBCXX_USE_CXX11_ABI=0']

    # Security flags
    eca += get_sdl_cflags()
    ela += get_sdl_ldflags()

    if IS_WIN:
        libraries_plat = ['daal_core_dll']
    else:
        libraries_plat = ['daal_core', 'daal_thread']

    if IS_MAC:
        ela.append('-stdlib=libc++')
        ela.append("-Wl,-rpath,{}".format(daal_lib_dir))
    elif IS_WIN:
        ela.append('-IGNORE:4197')
    elif IS_LIN and not any(x in os.environ and '-g' in os.environ[x] for x in ['CPPFLAGS', 'CFLAGS', 'LDFLAGS']):
        ela.append('-s')

    exts = cythonize([Extension('_daal4py',
                                [os.path.abspath('src/daal4py.cpp'),
                                 os.path.abspath('build/daal4py_cpp.cpp'),
                                 os.path.abspath('build/daal4py_cy.pyx')]
                                + DIST_CPPS,
                                depends=glob.glob(jp(os.path.abspath('src'), '*.h')),
                                include_dirs=include_dir_plat + [np.get_include()],
                                extra_compile_args=eca,
                                extra_link_args=ela,
                                libraries=libraries_plat + MPI_LIBS,
                                library_dirs=DAAL_LIBDIRS,
                                language='c++'),
    ])
    if dpcpp:
        exts.extend(cythonize(Extension('_oneapi',
                                        [os.path.abspath('src/oneapi/oneapi.pyx'),],
                                        depends=['src/oneapi/oneapi.h',],
                                        include_dirs=include_dir_plat + [np.get_include()],
                                        extra_compile_args=eca + ['-fsycl'],
                                        extra_link_args=ela,
                                        libraries=libraries_plat + DPCPP_LIBS,
                                        library_dirs=DAAL_LIBDIRS + DPCPP_LIBDIRS,
                                        language='c++')))
    if not no_dist:
        exts.append(Extension('mpi_transceiver',
                              MPI_CPPS,
                              depends=glob.glob(jp(os.path.abspath('src'), '*.h')),
                              include_dirs=include_dir_plat + [np.get_include()] + MPI_INCDIRS,
                              extra_compile_args=eca,
                              extra_link_args=ela + ["-Wl,-rpath,{}".format(x) for x in MPI_LIBDIRS],
                              libraries=libraries_plat + MPI_LIBS,
                              library_dirs=DAAL_LIBDIRS + MPI_LIBDIRS,
                              language='c++'))
    return exts


cfg_vars = get_config_vars()
for key, value in get_config_vars().items():
    if type(value) == str:
        cfg_vars[key] = value.replace("-Wstrict-prototypes", "").replace('-DNDEBUG', '')


def gen_pyx(odir):
    gtr_files = glob.glob(jp(os.path.abspath('generator'), '*')) + ['./setup.py']
    src_files = [os.path.abspath('build/daal4py_cpp.h'),
                 os.path.abspath('build/daal4py_cpp.cpp'),
                 os.path.abspath('build/daal4py_cy.pyx')]
    if all(os.path.isfile(x) for x in src_files):
        src_files.sort(key=lambda x: os.path.getmtime(x))
        gtr_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        if os.path.getmtime(src_files[0]) > os.path.getmtime(gtr_files[0]):
            print('Generated files are all newer than generator code. Skipping code generation')
            return

    from generator.gen_daal4py import gen_daal4py
    odir = os.path.abspath(odir)
    if not os.path.isdir(odir):
        os.mkdir(odir)
    gen_daal4py(daal_root, odir, d4p_version, no_dist=no_dist, no_stream=no_stream)


gen_pyx(os.path.abspath('./build'))


# daal setup
setup(  name        = "daal4py",
        description = "Convenient Python API to Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL)",
        author      = "Intel",
        version     = d4p_version,
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Environment :: Console',
            'Intended Audience :: Developers',
            'Intended Audience :: Other Audience',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: MacOS :: MacOS X',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX :: Linux',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Topic :: Scientific/Engineering',
            'Topic :: System',
            'Topic :: Software Development',
          ],
        setup_requires = ['numpy>=1.14', 'cython', 'jinja2'],
        packages = ['daal4py',
                    'daal4py.oneapi',
                    'daal4py.sklearn',
                    'daal4py.sklearn.cluster',
                    'daal4py.sklearn.decomposition',
                    'daal4py.sklearn.ensemble',
                    'daal4py.sklearn.linear_model',
                    'daal4py.sklearn.neighbors',
                    'daal4py.sklearn.monkeypatch',
                    'daal4py.sklearn.svm',
                    'daal4py.sklearn.utils',
        ],
        ext_modules = getpyexts(),
)
