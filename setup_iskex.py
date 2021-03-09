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
import sysconfig
import time
import subprocess
from setuptools import setup, Extension
import setuptools.command.install as orig_install
import setuptools.command.develop as orig_develop
import distutils.command.build as orig_build
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

if 'linux' in sys.platform:
    IS_LIN = True
elif sys.platform == 'darwin':
    IS_MAC = True
elif sys.platform in ['win32', 'cygwin']:
    IS_WIN = True
else:
    assert False, sys.platform + ' not supported'


iskex_version = (os.environ['ISKEX_VERSION'] if 'ISKEX_VERSION' in os.environ
               else time.strftime('2021.%Y%m%d.%H%M%S'))

trues = ['true', 'True', 'TRUE', '1', 't', 'T', 'y', 'Y', 'Yes', 'yes', 'YES']

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

# iskex setup
setup(name="iskex",
      description="Intel(R) Extension for scikit-learn with helps Intel(R) oneAPI Data Analytics Library",
      long_description=long_description,
      long_description_content_type="text/markdown",
      license="Apache-2.0",
      author="Intel Corporation",
      version=iskex_version,
      url='https://github.com/IntelPython/daal4py',
      author_email="scripting@intel.com",
      maintainer_email="onedal.maintainers@intel.com",
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
      packages=['iskex'],
#      packages=['daal4py',
#                'daal4py.oneapi',
#                'daal4py.sklearn',
#                'daal4py.sklearn.cluster',
#                'daal4py.sklearn.decomposition',
#                'daal4py.sklearn.ensemble',
#                'daal4py.sklearn.linear_model',
#                'daal4py.sklearn.manifold',
#                'daal4py.sklearn.metrics',
#                'daal4py.sklearn.neighbors',
#                'daal4py.sklearn.monkeypatch',
#                'daal4py.sklearn.svm',
#                'daal4py.sklearn.utils',
#                'daal4py.sklearn.model_selection',
#                ],
#      package_data={'daal4py.oneapi': ['liboneapi_backend.so',
#                                       'oneapi_backend.lib',
#                                       'oneapi_backend.dll'
#                                       ]
#                    },
#      ext_modules=getpyexts()
      )
