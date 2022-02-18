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

# System imports
import os
import time
from setuptools import setup
from scripts.package_helpers import get_packages_with_tests

sklearnex_version = (os.environ["SKLEARNEX_VERSION"] if "SKLEARNEX_VERSION" in os.environ
                     else time.strftime("2021.%Y%m%d.%H%M%S"))

project_urls = {
    "Bug Tracker": "https://github.com/intel/scikit-learn-intelex/issues",
    "Documentation": "https://intel.github.io/scikit-learn-intelex/",
    "Source Code": "https://github.com/intel/scikit-learn-intelex"
}

with open("README.md", "r", encoding="utf8") as f:
    long_description = f.read()


# sklearnex setup
setup(name="scikit-learn-intelex",
      description="Intel(R) Extension for Scikit-learn is a "
                  "seamless way to speed up your Scikit-learn application.",
      long_description=long_description,
      long_description_content_type="text/markdown",
      license="Apache-2.0",
      author="Intel Corporation",
      version=sklearnex_version,
      url="https://github.com/IntelPython/daal4py",
      author_email="scripting@intel.com",
      maintainer_email="onedal.maintainers@intel.com",
      project_urls=project_urls,
      classifiers=[
          "Development Status :: 5 - Production/Stable",
          "Environment :: Console",
          "Intended Audience :: Developers",
          "Intended Audience :: Other Audience",
          "Intended Audience :: Science/Research",
          "License :: OSI Approved :: Apache Software License",
          "Operating System :: MacOS :: MacOS X",
          "Operating System :: Microsoft :: Windows",
          "Operating System :: POSIX :: Linux",
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
          "Topic :: Scientific/Engineering",
          "Topic :: System",
          "Topic :: Software Development",
      ],
      python_requires=">=3.6",
      install_requires=[
          "daal4py>=2021.2",
          "scikit-learn>=0.22"
      ],
      keywords=[
          "machine learning",
          "scikit-learn",
          "data science",
          "data analytics",
      ],
      packages=get_packages_with_tests([
          "sklearnex",
          'sklearnex.cluster',
          'sklearnex.decomposition',
          'sklearnex.ensemble',
          'sklearnex.glob',
          'sklearnex.linear_model',
          'sklearnex.manifold',
          'sklearnex.metrics',
          'sklearnex.model_selection',
          'sklearnex.neighbors',
          'sklearnex.svm',
          'sklearnex.utils'
      ]),
      )
