#!/usr/bin/env python
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


def get_patch_str():
    return \
"""try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    del patch_sklearn
except (ImportError, ModuleNotFoundError):
    pass"""


def global_patching():
    try:
        import sklearn
    except ImportError:
        # TODO: change raise to another error handling?
        raise ImportError("Scikit-learn could not be imported. Nothing to patch\n")

    init_file_path = sklearn.__file__
    distributor_file_path = os.path.join(os.path.dirname(init_file_path), "_distributor_init.py")
    
    with open(distributor_file_path,'r',encoding = 'utf-8') as distributor_file:
        if get_patch_str() in distributor_file.read():
            print("Scikit-learn already patched")
            exit(0)
    
    with open(distributor_file_path,'a',encoding = 'utf-8') as distributor_file:
        distributor_file.write("\n" + get_patch_str() + "\n")


def global_unpatching():
    try:
        import sklearn
    except ImportError:
        # TODO: change raise to another error handling?
        raise ImportError("Scikit-learn could not be imported. Nothing to unpatch\n")

    init_file_path = sklearn.__file__
    distributor_file_path = os.path.join(os.path.dirname(init_file_path), "_distributor_init.py")
    
    with open(distributor_file_path,'r',encoding = 'utf-8') as distributor_file:
        lines = distributor_file.read()
        lines = lines.replace("\n" + get_patch_str(), '')

    with open(distributor_file_path,'w',encoding = 'utf-8') as distributor_file:
        distributor_file.write(lines)
