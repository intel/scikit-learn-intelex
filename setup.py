#! /usr/bin/env python
#===============================================================================
# Copyright 2014 Intel Corporation
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
import setuptools.command.develop as orig_develop
import distutils.command.build as orig_build
from os.path import join as jp
from distutils.sysconfig import get_config_vars
from Cython.Build import cythonize
import glob
import numpy as np
from scripts.version import get_onedal_version
import scripts.build_backend as build_backend
from scripts.package_helpers import get_packages_with_tests

try:
    from ctypes.utils import find_library
except ImportError:
    from ctypes.util import find_library

IS_WIN = False
IS_MAC = False
IS_LIN = False

dal_root = os.environ.get('DALROOT')

if dal_root is None:
    raise RuntimeError("Not set DALROOT variable")

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

ONEDAL_VERSION = get_onedal_version(dal_root)
ONEDAL_2021_3 = 2021 * 10000 + 3 * 100
is_onedal_iface = \
    os.environ.get('OFF_ONEDAL_IFACE') is None and ONEDAL_VERSION >= ONEDAL_2021_3


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
no_stream = 'NO_STREAM' in os.environ and os.environ['NO_STREAM'] in trues
mpi_root = None if no_dist else os.environ['MPIROOT']
dpcpp = True if 'DPCPPROOT' in os.environ else False
dpcpp_root = None if not dpcpp else os.environ['DPCPPROOT']
dpctl = True if dpcpp and 'DPCTLROOT' in os.environ else False
dpctl_root = None if not dpctl else os.environ['DPCTLROOT']


daal_lib_dir = lib_dir if (IS_MAC or os.path.isdir(
    lib_dir)) else os.path.dirname(lib_dir)
ONEDAL_LIBDIRS = [daal_lib_dir]
if IS_WIN:
    ONEDAL_LIBDIRS.append(f"{os.environ.get('CONDA_PREFIX')}/Library/lib")

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


def get_sdl_cflags():
    if IS_LIN or IS_MAC:
        if dpcpp:
            DIST_CFLAGS.extend(['-fsycl'])
        DIST_CFLAGS.extend(['-fstack-protector-strong', '-fPIC',
                            '-D_FORTIFY_SOURCE=2', '-Wformat',
                            '-Wformat-security', '-fno-strict-overflow',
                            '-fno-delete-null-pointer-checks'])
        return DIST_CFLAGS
    if IS_WIN:
        return DIST_CFLAGS + ['-GS', '-fsycl']


def get_sdl_ldflags():
    if IS_LIN:
        return ['-Wl,-z,noexecstack,-z,relro,-z,now,-fstack-protector-strong,'
                '-fno-strict-overflow,-fno-delete-null-pointer-checks,-fwrapv']
    if IS_MAC:
        return ['-fstack-protector-strong',
                '-fno-strict-overflow',
                '-fno-delete-null-pointer-checks',
                '-fwrapv']
    if IS_WIN:
        return ['-NXCompat', '-DynamicBase']


def get_daal_type_defines():
    daal_type_defines = ['DAAL_ALGORITHM_FP_TYPE',
                         'DAAL_SUMMARY_STATISTICS_TYPE',
                         'DAAL_DATA_TYPE']
    return [(d, 'double') for d in daal_type_defines]


def get_libs(iface='daal'):
    if IS_WIN:
        major_version = get_win_major_version()
        libraries_plat = [f'onedal_core_dll{major_version}']
        onedal_lib = [f'onedal_dll{major_version}']
        onedal_dpc_lib = [f'onedal_dpc_dll{major_version}']
    else:
        libraries_plat = ['onedal_core', 'onedal_thread']
        onedal_lib = ['onedal']
        onedal_dpc_lib = ['onedal_dpc']
    if iface == 'onedal':
        libraries_plat = onedal_lib + libraries_plat
    elif iface == 'onedal_dpc':
        libraries_plat = onedal_dpc_lib + libraries_plat
    return libraries_plat


def get_build_options():
    include_dir_plat = [os.path.abspath('./src'),
                        os.path.abspath('.'),
                        dal_root + '/include', ]
    # FIXME it is a wrong place for this dependency
    if not no_dist:
        include_dir_plat.append(mpi_root + '/include')
    using_intel = os.environ.get('cc', '') in [
        'icc', 'icpc', 'icl', 'dpcpp', 'icx', 'icpx']
    eca = ['-DPY_ARRAY_UNIQUE_SYMBOL=daal4py_array_API',
           '-DD4P_VERSION="' + d4p_version + '"', '-DNPY_ALLOW_THREADS=1']
    ela = []

    if using_intel and IS_WIN:
        include_dir_plat.append(
            jp(os.environ.get('ICPP_COMPILER16', ''), 'compiler', 'include'))
        eca += ['-std=c++17', '-w', '/MD', '-fsycl']
    elif not using_intel and IS_WIN:
        eca += ['-wd4267', '-wd4244', '-wd4101', '-wd4996', '/std:c++17']
    else:
        eca += ['-std=c++17', '-w', ]  # '-D_GLIBCXX_USE_CXX11_ABI=0']

    # Security flags
    eca += get_sdl_cflags()
    ela += get_sdl_ldflags()

    if IS_MAC:
        eca.append('-stdlib=libc++')
        ela.append('-stdlib=libc++')
        ela.append("-Wl,-rpath,{}".format(daal_lib_dir))
        ela.append("-Wl,-rpath,@loader_path/../../../")
    elif IS_WIN:
        ela.append('-IGNORE:4197')
    elif IS_LIN and not any(x in os.environ and '-g' in os.environ[x]
                            for x in ['CPPFLAGS', 'CFLAGS', 'LDFLAGS']):
        ela.append('-s')
    if IS_LIN:
        ela.append("-fPIC")
        ela.append("-Wl,-rpath,$ORIGIN/../../../")
    return eca, ela, include_dir_plat


def getpyexts():
    eca, ela, include_dir_plat = get_build_options()
    libraries_plat = get_libs("daal")

    exts = []

    ext = Extension('daal4py._daal4py',
                    [os.path.abspath('src/daal4py.cpp'),
                     os.path.abspath('build/daal4py_cpp.cpp'),
                     os.path.abspath('build/daal4py_cy.pyx')] + DIST_CPPS,
                    depends=glob.glob(jp(os.path.abspath('src'), '*.h')),
                    include_dirs=include_dir_plat + [np.get_include()],
                    extra_compile_args=eca,
                    define_macros=get_daal_type_defines(),
                    extra_link_args=ela,
                    libraries=libraries_plat,
                    library_dirs=ONEDAL_LIBDIRS,
                    language='c++')
    exts.extend(cythonize(ext))

    if dpcpp:
        if IS_LIN or IS_MAC:
            runtime_oneapi_dirs = ["$ORIGIN/oneapi"]
        elif IS_WIN:
            runtime_oneapi_dirs = []

        ext = Extension('daal4py._oneapi',
                        [os.path.abspath('src/oneapi/oneapi.pyx'), ],
                        depends=['src/oneapi/oneapi.h', 'src/oneapi/oneapi_backend.h'],
                        include_dirs=include_dir_plat + [np.get_include()],
                        extra_compile_args=eca,
                        extra_link_args=ela,
                        define_macros=[
                            ('NPY_NO_DEPRECATED_API',
                             'NPY_1_7_API_VERSION')
                        ],
                        libraries=['oneapi_backend'] + libraries_plat,
                        library_dirs=['daal4py/oneapi'] + ONEDAL_LIBDIRS,
                        runtime_library_dirs=runtime_oneapi_dirs,
                        language='c++')
        exts.extend(cythonize(ext))

    if not no_dist:
        mpi_include_dir = include_dir_plat + [np.get_include()] + MPI_INCDIRS
        mpi_depens = glob.glob(jp(os.path.abspath('src'), '*.h'))
        mpi_extra_link = ela + ["-Wl,-rpath,{}".format(x) for x in MPI_LIBDIRS]
        exts.append(Extension('daal4py.mpi_transceiver',
                              MPI_CPPS,
                              depends=mpi_depens,
                              include_dirs=mpi_include_dir,
                              extra_compile_args=eca,
                              define_macros=get_daal_type_defines(),
                              extra_link_args=mpi_extra_link,
                              libraries=libraries_plat + MPI_LIBS,
                              library_dirs=ONEDAL_LIBDIRS + MPI_LIBDIRS,
                              language='c++'))
    return exts


cfg_vars = get_config_vars()
for key, value in get_config_vars().items():
    if isinstance(value, str):
        cfg_vars[key] = value.replace(
            "-Wstrict-prototypes", "").replace('-DNDEBUG', '')


def gen_pyx(odir):
    gtr_files = glob.glob(
        jp(os.path.abspath('generator'), '*')) + ['./setup.py']
    src_files = [os.path.abspath('build/daal4py_cpp.h'),
                 os.path.abspath('build/daal4py_cpp.cpp'),
                 os.path.abspath('build/daal4py_cy.pyx')]
    if all(os.path.isfile(x) for x in src_files):
        src_files.sort(key=os.path.getmtime)
        gtr_files.sort(key=os.path.getmtime, reverse=True)
        if os.path.getmtime(src_files[0]) > os.path.getmtime(gtr_files[0]):
            print('Generated files are all newer than generator code.'
                  'Skipping code generation')
            return

    from generator.gen_daal4py import gen_daal4py
    odir = os.path.abspath(odir)
    if not os.path.isdir(odir):
        os.mkdir(odir)
    gen_daal4py(dal_root, odir, d4p_version,
                no_dist=no_dist, no_stream=no_stream)


gen_pyx(os.path.abspath('./build'))


def build_oneapi_backend():
    eca, ela, includes = get_build_options()

    return build_backend.build_cpp(
        cc='icx',
        cxx='icx',
        sources=['src/oneapi/oneapi_backend.cpp'],
        targetname='oneapi_backend',
        targetprefix='' if IS_WIN else 'lib',
        targetsuffix='.dll' if IS_WIN else '.so',
        libs=get_libs('daal') + ['OpenCL', 'onedal_sycl'],
        libdirs=ONEDAL_LIBDIRS,
        includes=includes,
        eca=eca,
        ela=ela,
        defines=[],
        installpath='daal4py/oneapi/'
    )


def get_onedal_py_libs():
    ext_suffix = get_config_vars('EXT_SUFFIX')[0]
    libs = [f'_onedal_py_host{ext_suffix}', f'_onedal_py_dpc{ext_suffix}']
    if IS_WIN:
        ext_suffix_lib = ext_suffix.replace('.dll', '.lib')
        libs += [f'_onedal_py_host{ext_suffix_lib}', f'_onedal_py_dpc{ext_suffix_lib}']
    return libs


class custom_build():
    def run(self):
        if is_onedal_iface:
            cxx = os.getenv('CXX', 'cl' if IS_WIN else 'g++')
            build_backend.custom_build_cmake_clib('host', cxx)
        if dpcpp:
            build_oneapi_backend()
            if is_onedal_iface:
                build_backend.custom_build_cmake_clib('dpc')


class develop(orig_develop.develop, custom_build):
    def run(self):
        custom_build.run(self)
        return super().run()


class build(orig_build.build, custom_build):
    def run(self):
        custom_build.run(self)
        return super().run()


project_urls = {
    'Bug Tracker': 'https://github.com/IntelPython/daal4py/issues',
    'Documentation': 'https://intelpython.github.io/daal4py/',
    'Source Code': 'https://github.com/IntelPython/daal4py'
}

with open('README.md', 'r', encoding='utf8') as f:
    long_description = f.read()

install_requires = []
with open('requirements.txt') as f:
    install_requires.extend(f.read().splitlines())
    if IS_MAC:
        for r in install_requires:
            if "dpcpp_cpp_rt" in r:
                install_requires.remove(r)
                break

setup(
    name="daal4py",
    description="A convenient Python API to Intel(R) oneAPI Data Analytics Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    author="Intel Corporation",
    version=d4p_version,
    url='https://github.com/IntelPython/daal4py',
    author_email="scripting@intel.com",
    maintainer_email="onedal.maintainers@intel.com",
    project_urls=project_urls,
    cmdclass={'develop': develop, 'build': build},
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
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
        'Topic :: System',
        'Topic :: Software Development',
    ],
    python_requires='>=3.6',
    install_requires=install_requires,
    keywords=[
        'machine learning',
        'scikit-learn',
        'data science',
        'data analytics'
    ],
    packages=get_packages_with_tests([
        'daal4py',
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
        'onedal',
        'onedal.svm',
        'onedal.neighbors',
        'onedal.primitives',
        'onedal.datatypes',
        'onedal.common'
    ]),
    package_data={
        'daal4py.oneapi': [
            'liboneapi_backend.so',
            'oneapi_backend.lib',
            'oneapi_backend.dll',
        ],
        'onedal': get_onedal_py_libs()
    },
    ext_modules=getpyexts()
)
