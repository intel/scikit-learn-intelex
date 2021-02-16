#! /usr/bin/env python
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

# System imports
import os
import sys
import time
from setuptools import setup, Extension
from os.path import join as jp
from distutils.sysconfig import get_config_vars
from Cython.Build import cythonize
import glob
import numpy as np
import distutils.ccompiler

try:
    from ctypes.utils import find_library
except ImportError:
    from ctypes.util import find_library

IS_WIN = False
IS_MAC = False
IS_LIN = False

daal_root = os.environ.get('DAALROOT')
dal_root = os.environ.get('DALROOT')
if not dal_root:
    dal_root = daal_root

if 'linux' in sys.platform:
    IS_LIN = True
    lib_dir = jp(dal_root, 'lib', 'intel64')
elif sys.platform == 'darwin':
    IS_MAC = True
    lib_dir = jp(dal_root, 'lib')
elif sys.platform in ['win32', 'cygwin']:
    IS_WIN = True
    lib_dir = jp(dal_root, 'lib', 'intel64')
    if sys.platform == "win32":
        # noinspection PyUnresolvedReferences
        import dpcppcompiler
        sys.modules["distutils.dpcppcompiler"] = sys.modules["dpcppcompiler"]
        distutils.ccompiler.compiler_class["clang-cl"] = (
            "dpcppcompiler", "DPCPPCompiler", "Support of DPCPP compiler"
        )
else:
    assert False, sys.platform + ' not supported'


def get_lib_suffix():

    def walk_ld_library_path():
        if IS_WIN:
            ld_library_path = os.environ.get('LIBRARY_LIB')
            if ld_library_path is None:
                ld_library_path = f"{os.environ.get('CONDA_PREFIX')}/Library/lib"
        else:
            ld_library_path = os.environ.get('LD_LIBRARY_PATH', None)

        if ld_library_path is None:
            return None

        libs = []
        if IS_WIN:
            ld_library_path = ld_library_path.split(';')
        else:
            ld_library_path = ld_library_path.split(':')
        while '' in ld_library_path:
            ld_library_path.remove('')
        for lib_path in ld_library_path:
            for _, _, new_files in os.walk(lib_path):
                libs += new_files

        for lib in libs:
            if 'onedal_core' in lib:
                return 'onedal'
            if 'daal_core' in lib:
                return 'daal'
        return None

    def walk_libdir():
        global lib_dir

        for _, _, libs in os.walk(lib_dir):
            for lib in libs:
                if 'onedal_core' in lib:
                    return 'onedal'
                if 'daal_core' in lib:
                    return 'daal'
        return None

    ld_lib_path_suffix = walk_ld_library_path()
    lib_dir_suffix = walk_libdir()
    if find_library('onedal_core') is not None or \
       ld_lib_path_suffix == 'onedal' or \
            lib_dir_suffix == 'onedal':
        return 'onedal'
    if find_library('daal_core') is not None or \
       ld_lib_path_suffix == 'daal' or \
            lib_dir_suffix == 'daal':
        return 'daal'

    raise ImportError('Unable to import oneDAL or oneDAL lib')


def get_win_major_version():
    lib_name = find_library('onedal_core')
    if lib_name is None:
        return ''
    version = lib_name.split('\\')[-1].split('.')[1]
    try:
        version = '.' + str(int(version))
    except ValueError:
        version = ''
    return version


d4p_version = (os.environ['DAAL4PY_VERSION'] if 'DAAL4PY_VERSION' in os.environ
               else time.strftime('2021.%Y%m%d.%H%M%S'))

trues = ['true', 'True', 'TRUE', '1', 't', 'T', 'y', 'Y', 'Yes', 'yes', 'YES']
no_dist = True if 'NO_DIST' in os.environ and os.environ['NO_DIST'] in trues else False
if not no_dist and sys.version_info <= (3, 6):
    print('distributed mode not supported for python version < 3.6\n')
    no_dist = True
no_stream = 'NO_STREAM' in os.environ and os.environ['NO_STREAM'] in trues
mpi_root = None if no_dist else os.environ['MPIROOT']
dpcpp = True if 'DPCPPROOT' in os.environ else False
dpcpp_root = None if not dpcpp else os.environ['DPCPPROOT']
dpctl = True if dpcpp and 'DPCTLROOT' in os.environ else False
dpctl_root = None if not dpctl else os.environ['DPCTLROOT']

#itac_root = os.environ['VT_ROOT']
IS_WIN = False
IS_MAC = False
IS_LIN = False

if 'linux' in sys.platform:
    IS_LIN = True
    lib_dir = jp(dal_root, 'lib', 'intel64')
elif sys.platform == 'darwin':
    IS_MAC = True
    lib_dir = jp(dal_root, 'lib')
elif sys.platform in ['win32', 'cygwin']:
    IS_WIN = True
    lib_dir = jp(dal_root, 'lib', 'intel64')
else:
    assert False, sys.platform + ' not supported'

daal_lib_dir = lib_dir if (IS_MAC or os.path.isdir(lib_dir)) else os.path.dirname(lib_dir)
DAAL_LIBDIRS = [daal_lib_dir]
if IS_WIN:
    DAAL_LIBDIRS.append(f"{os.environ.get('CONDA_PREFIX')}/Library/lib")

if no_stream:
    print('\nDisabling support for streaming mode\n')
if no_dist:
    print('\nDisabling support for distributed mode\n')
    DIST_CFLAGS = []
    DIST_CPPS = []
    MPI_INCDIRS = []
    MPI_LIBDIRS = []
    MPI_LIBS = []
    MPI_CPPS = []
else:
    DIST_CFLAGS = ['-D_DIST_', ]
    DIST_CPPS = ['src/transceiver.cpp']
    MPI_INCDIRS = [jp(mpi_root, 'include')]
    MPI_LIBDIRS = [jp(mpi_root, 'lib')]
    MPI_LIBNAME = getattr(os.environ, 'MPI_LIBNAME', None)
    if MPI_LIBNAME:
        MPI_LIBS = [MPI_LIBNAME]
    elif IS_WIN:
        if os.path.isfile(jp(mpi_root, 'lib', 'mpi.lib')):
            MPI_LIBS = ['mpi']
        if os.path.isfile(jp(mpi_root, 'lib', 'impi.lib')):
            MPI_LIBS = ['impi']
        assert MPI_LIBS, "Couldn't find MPI library"
    else:
        MPI_LIBS = ['mpi']
    MPI_CPPS = ['src/mpi/mpi_transceiver.cpp']

#Level Zero workaround for oneDAL Beta06
from generator.parse import parse_version

header_path = os.path.join(dal_root, 'include', 'services', 'library_version_info.h')

with open(header_path) as header:
    v = parse_version(header)
    dal_build_version = (int(v[0]), int(v[1]), int(v[2]), str(v[3]))

if dpcpp:
    DPCPP_CFLAGS = ['-D_DPCPP_ -fno-builtin-memset']
    DPCPP_LIBS = ['OpenCL', 'sycl', 'onedal_sycl']
    if IS_LIN:
        DPCPP_LIBDIRS = [jp(dpcpp_root, 'linux', 'lib')]
    elif IS_WIN:
        DPCPP_LIBDIRS = [jp(dpcpp_root, 'windows', 'lib')]

    if dpctl:
        # if custom dpctl library directory is specified
        if 'DPCTL_LIBPATH' in os.environ:
            DPCTL_LIBDIRS = [os.environ['DPCTL_LIBPATH']]
        else:
            DPCTL_LIBDIRS = [jp(dpctl_root, 'lib')]
        DPCTL_INCDIRS = [jp(dpctl_root, 'include')]
        DPCTL_LIBS = ['DPPLSyclInterface']
    else:
        DPCTL_INCDIRS = []
        DPCTL_LIBDIRS = []
        DPCTL_LIBS = []

else:
    DPCPP_CFLAGS = []
    DPCPP_LIBS = []
    DPCPP_LIBDIRS = []

DAAL_DEFAULT_TYPE = 'double'


def get_sdl_cflags():
    if IS_LIN or IS_MAC:
        return DIST_CFLAGS + DPCPP_CFLAGS + ['-fstack-protector-strong', '-fPIC',
                                             '-D_FORTIFY_SOURCE=2', '-Wformat',
                                             '-Wformat-security', '-fno-strict-overflow',
                                             '-fno-delete-null-pointer-checks']
    elif IS_WIN:
        return DIST_CFLAGS + DPCPP_CFLAGS + ['-GS', ]


def get_sdl_ldflags():
    if IS_LIN:
        return ['-Wl,-z,noexecstack,-z,relro,-z,now,-fstack-protector-strong,'
                '-fno-strict-overflow,-fno-delete-null-pointer-checks,-fwrapv']
    elif IS_MAC:
        return ['-fstack-protector-strong',
                '-fno-strict-overflow',
                '-fno-delete-null-pointer-checks',
                '-fwrapv']
    elif IS_WIN:
        return ['-NXCompat', '-DynamicBase']


def get_type_defines():
    daal_type_defines = ['DAAL_ALGORITHM_FP_TYPE',
                         'DAAL_SUMMARY_STATISTICS_TYPE',
                         'DAAL_DATA_TYPE']
    return ["-D{}={}".format(d, DAAL_DEFAULT_TYPE) for d in daal_type_defines]


def getpyexts():
    include_dir_plat = [os.path.abspath('./src'), dal_root + '/include', ]
    # FIXME it is a wrong place for this dependency
    if not no_dist:
        include_dir_plat.append(mpi_root + '/include')
    using_intel = os.environ.get('cc', '') in ['icc', 'icpc', 'icl', 'dpcpp']
    eca = ['-DPY_ARRAY_UNIQUE_SYMBOL=daal4py_array_API',
           '-DD4P_VERSION="' + d4p_version + '"',
           '-DNPY_ALLOW_THREADS=1'] + get_type_defines()
    ela = []

    if using_intel and IS_WIN:
        if os.environ.get('cc', '') == "dpcpp" or \
                os.environ.get('cc', '') == "clang++" or \
                os.environ.get('cc', '') == "clang-cl":
            eca.append("/EHsc")
        else:
            include_dir_plat.append(
                jp(os.environ.get('ICPP_COMPILER16', ''),
                   'compiler',
                   'include')
            )
            eca += ['-std=c++11', '-w', '/MD']
    elif not using_intel and IS_WIN:
        eca += ['-wd4267', '-wd4244', '-wd4101', '-wd4996', '/MD']
    else:
        eca += ['-std=c++11', '-w', ]  # '-D_GLIBCXX_USE_CXX11_ABI=0']

    # Security flags
    eca += get_sdl_cflags()
    ela += get_sdl_ldflags()

    lib_suffix = get_lib_suffix()

    if IS_WIN:
        major_version = get_win_major_version()
        libraries_plat = [f'{lib_suffix}_core_dll{major_version}']
    else:
        libraries_plat = [f'{lib_suffix}_core', f'{lib_suffix}_thread']

    if IS_MAC:
        ela.append('-stdlib=libc++')
        ela.append("-Wl,-rpath,{}".format(daal_lib_dir))
    elif IS_WIN:
        ela.append('-IGNORE:4197')

    elif IS_LIN and not any(x in os.environ and '-g' in os.environ[x]
                            for x in ['CPPFLAGS', 'CFLAGS', 'LDFLAGS']):
        ela.append('-s')

    exts = cythonize([Extension('_daal4py',
                                [os.path.abspath('src/daal4py.cpp'),
                                 os.path.abspath('build/daal4py_cpp.cpp'),
                                 os.path.abspath('build/daal4py_cy.pyx')] + DIST_CPPS,
                                depends=glob.glob(jp(os.path.abspath('src'), '*.h')),
                                include_dirs=include_dir_plat + [np.get_include()],
                                extra_compile_args=eca,
                                extra_link_args=ela,
                                libraries=libraries_plat,
                                library_dirs=DAAL_LIBDIRS,
                                language='c++'),
                      ])

    eca_dpcpp = eca.copy() + ['-fsycl']

    if dpcpp:
        ext = Extension('_oneapi',
                        [os.path.abspath('src/oneapi/oneapi.pyx'), ],
                        depends=['src/oneapi/oneapi.h', ],
                        include_dirs=include_dir_plat + [np.get_include()],
                        extra_compile_args=eca_dpcpp,
                        extra_link_args=ela,
                        libraries=libraries_plat + DPCPP_LIBS,
                        library_dirs=DAAL_LIBDIRS + DPCPP_LIBDIRS,
                        language='c++')
        exts.extend(cythonize(ext))
    if dpctl:
        ext = Extension('_dpctl_interop',
                        [
                            os.path.abspath('src/dpctl_interop/dpctl_interop.pyx'),
                            os.path.abspath('src/dpctl_interop/daal_context_service.cpp'),
                        ],
                        depends=['src/dpctl_interop/daal_context_service.h', ],
                        include_dirs=include_dir_plat + DPCTL_INCDIRS,
                        extra_compile_args=eca_dpcpp,
                        extra_link_args=ela,
                        libraries=libraries_plat + DPCPP_LIBS + DPCTL_LIBS,
                        library_dirs=DAAL_LIBDIRS + DPCPP_LIBDIRS + DPCTL_LIBDIRS,
                        language='c++')
        exts.extend(cythonize(ext))

    if not no_dist:
        ext = Extension('mpi_transceiver',
                        MPI_CPPS,
                        depends=glob.glob(jp(os.path.abspath('src'), '*.h')),
                        include_dirs=include_dir_plat + [np.get_include()] + MPI_INCDIRS,
                        extra_compile_args=eca,
                        extra_link_args=ela + ["-Wl,-rpath,{}".format(x)
                                               for x in MPI_LIBDIRS],
                        libraries=libraries_plat + MPI_LIBS,
                        library_dirs=DAAL_LIBDIRS + MPI_LIBDIRS,
                        language='c++')
        exts.append(ext)
    return exts


cfg_vars = get_config_vars()
for key, value in get_config_vars().items():
    if isinstance(value, str):
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
            print('Generated files are all newer than generator code.'
                  'Skipping code generation')
            return

    from generator.gen_daal4py import gen_daal4py
    odir = os.path.abspath(odir)
    if not os.path.isdir(odir):
        os.mkdir(odir)
    gen_daal4py(dal_root, odir, d4p_version, no_dist=no_dist, no_stream=no_stream)


gen_pyx(os.path.abspath('./build'))

project_urls = {
    'Bug Tracker': 'https://github.com/IntelPython/daal4py/issues',
    'Documentation': 'https://intelpython.github.io/daal4py/',
    'Source Code': 'https://github.com/IntelPython/daal4py'
}

# daal setup
setup(name="daal4py",
      description="A convenient Python API to Intel(R) oneAPI Data Analytics Library",
      author="Intel",
      version=d4p_version,
      url='https://github.com/IntelPython/daal4py',
      project_urls=project_urls,
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
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering',
          'Topic :: System',
          'Topic :: Software Development',
      ],
      setup_requires=['numpy>=1.14', 'cython', 'jinja2'],
      install_requires=['numpy>=1.14', 'daal', 'dpcpp_cpp_rt'],
      packages=['daal4py',
                'daal4py.oneapi',
                'daal4py.sklearn',
                'daal4py.sklearn.cluster',
                'daal4py.sklearn.decomposition',
                'daal4py.sklearn.ensemble',
                'daal4py.sklearn.linear_model',
                'daal4py.sklearn.manifold',
                'daal4py.sklearn.metrics',
                'daal4py.sklearn.neighbors',
                'daal4py.sklearn.monkeypatch',
                'daal4py.sklearn.svm',
                'daal4py.sklearn.utils',
                'daal4py.sklearn.model_selection',
                ],
      ext_modules=getpyexts()
      )
