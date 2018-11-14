#! /usr/bin/env python
#*******************************************************************************
# Copyright 2014-2017 Intel Corporation
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

import numpy as np

npyver = int(np.__version__.split('.')[1])

if npyver == 9:
    print("Warning:  Detected numpy version {}".format(np.__version__))
    print("Numpy 1.10 or greater is strongly recommended.")
    print("Earlier versions have not been tested. Use at your own risk.")

if npyver < 9:
    sys.exit("Error: Detected numpy {}. The minimum requirement is 1.9, and >= 1.10 is strongly recommended".format(np.__version__))

d4p_version = os.environ['DAAL4PY_VERSION'] if 'DAAL4PY_VERSION' in os.environ else time.strftime('0.2019.%Y%m%d.%H%M%S')
no_dist = True if 'NO_DIST' in os.environ and os.environ['NO_DIST'] in ['true', 'True', 'TRUE', '1', 't', 'T', 'y', 'Y', 'Yes', 'yes', 'YES'] else False
daal_root = os.environ['DAALROOT']
tbb_root = os.environ['TBBROOT']
mpi_root = None if no_dist else os.environ['MPIROOT']

#itac_root = os.environ['VT_ROOT']
IS_WIN = False
IS_MAC = False
IS_LIN = False

if 'linux' in sys.platform:
    IS_LIN = True
    lib_dir = jp(daal_root, 'lib', 'intel64_lin')
elif sys.platform == 'darwin':
    IS_MAC = True
    lib_dir = jp(daal_root, 'lib')
elif sys.platform in ['win32', 'cygwin']:
    IS_WIN = True
    lib_dir = jp(daal_root, 'lib', 'intel64_win')
else:
    assert False, sys.platform + ' not supported'

daal_lib_dir = lib_dir if (IS_MAC or os.path.isdir(lib_dir)) else os.path.dirname(lib_dir)


if no_dist :
    print('\nDisabling support for distributed mode\n')
    DIST_CFLAGS  = []
    DIST_INCDIRS = []
    DIST_LIBDIRS = []
    DIST_LIBS    = []
else:
    DIST_CFLAGS  = ['-D_DIST_',]
    DIST_INCDIRS = [jp(mpi_root, 'include')]
    DIST_LIBDIRS = [jp(mpi_root, 'lib')]
    if IS_WIN:
        if os.path.isfile(jp(mpi_root, 'lib', 'mpi.lib')):
            DIST_LIBS    = ['mpi']
        if os.path.isfile(jp(mpi_root, 'lib', 'impi.lib')):
            DIST_LIBS    = ['impi']
        assert DIST_LIBS, "Couldn't find MPI library"
    else:
        DIST_LIBS    = ['mpi']
DAAL_DEFAULT_TYPE = 'double'

def get_sdl_cflags():
    if IS_LIN or IS_MAC:
        return DIST_CFLAGS + ['-fstack-protector', '-fPIC',
                              '-D_FORTIFY_SOURCE=2', '-Wformat', '-Wformat-security',]
    elif IS_WIN:
        return DIST_CFLAGS + ['-GS',]

def get_sdl_ldflags():
    if IS_LIN:
        return ['-Wl,-z,noexecstack', '-Wl,-z,relro', '-Wl,-z,now',]
    elif IS_MAC:
        return []
    elif IS_WIN:
        return ['-NXCompat', '-DynamicBase']

def get_type_defines():
    daal_type_defines = ['DAAL_ALGORITHM_FP_TYPE', 'DAAL_SUMMARY_STATISTICS_TYPE', 'DAAL_DATA_TYPE']
    return ["-D{}={}".format(d, DAAL_DEFAULT_TYPE) for d in daal_type_defines]

def getpyexts():
    include_dir_plat = set([os.path.abspath('./src'), daal_root + '/include', tbb_root + '/include',] + DIST_INCDIRS)
    using_intel = os.environ.get('cc', '') in ['icc', 'icpc', 'icl']
    eca = ['-DPY_ARRAY_UNIQUE_SYMBOL=daal4py_array_API', '-DD4P_VERSION="'+d4p_version+'"'] + get_type_defines()
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

    if sys.version_info[0] >= 3:
        eca.append('-DUSE_CAPSULE')

    if IS_WIN:
        libraries_plat = ['daal_core_dll']
    else:
        libraries_plat = ['daal_core', 'daal_thread']
    libraries_plat += DIST_LIBS

    if IS_MAC:
        ela.append('-stdlib=libc++')
        ela.append("-Wl,-rpath,{}".format(daal_lib_dir))
        for x in DIST_LIBDIRS:
            ela.append("-Wl,-rpath,{}".format(x))
        ela.append("-Wl,-rpath,{}".format(jp(daal_root, '..', 'tbb', 'lib')))
    elif IS_WIN:
        ela.append('-IGNORE:4197')
    elif IS_LIN and not any(x in os.environ and '-g' in os.environ[x] for x in ['CPPFLAGS', 'CFLAGS', 'LDFLAGS']):
        ela.append('-s')

    return cythonize([Extension('_daal4py',
                                [os.path.abspath('src/daal4py.cpp'),
                                 os.path.abspath('build/daal4py_cpp.cpp'),
                                 os.path.abspath('build/daal4py_cy.pyx')],
                                include_dirs=include_dir_plat.union([np.get_include()]),
                                extra_compile_args=eca,
                                extra_link_args=ela,
                                libraries=libraries_plat,
                                library_dirs=[daal_lib_dir] + DIST_LIBDIRS,
                                language='c++')])

cfg_vars = get_config_vars()
for key, value in get_config_vars().items():
    if type(value) == str:
        cfg_vars[key] = value.replace("-Wstrict-prototypes", "").replace('-DNDEBUG', '')


from generator.gen_daal4py import gen_daal4py
def gen_pyx(odir):
    odir = os.path.abspath(odir)
    if not os.path.isdir(odir):
        os.mkdir(odir)
    gen_daal4py(daal_root, odir, d4p_version)

gen_pyx(os.path.abspath('./build'))

# daal setup
setup(  name        = "daal4py",
        description = "Higher Level Python API to Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL)",
        author      = "Intel",
        version     = d4p_version,
        classifiers=[
            'Development Status :: 4 - ALPHA',
            'Environment :: Console',
            'Intended Audience :: Developers',
            'Intended Audience :: System Administrators',
            'Intended Audience :: Other Audience',
            'Intended Audience :: Science/Research',
            'License :: Other/Proprietary License',
            'Operating System :: MacOS :: MacOS X',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX :: Linux',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Topic :: Scientific/Engineering',
            'Topic :: System',
            'Topic :: Software Development',
          ],
        setup_requires = ['numpy>=1.11', 'cython'],
        packages = ['daal4py'],
        ext_modules = getpyexts(),
)
