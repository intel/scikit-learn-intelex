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

import re
from os.path import join as jp


def get_onedal_version(dal_root):
    """Parse oneDAL version strings"""

    header_version = jp(dal_root, 'include', 'services', 'library_version_info.h')
    version = ""

    major, minnor = "", ""
    with open(header_version, 'r') as header:
        for elem in header:
            if '#define __INTEL_DAAL__' in elem:
                match = re.match(r'#define __INTEL_DAAL__ (\d+)', elem)
                if match:
                    major = match.group(1)

            if '#define __INTEL_DAAL_MINOR__' in elem:
                match = re.match(r'#define __INTEL_DAAL_MINOR__ (\d+)', elem)
                if match:
                    minnor = match.group(1)
    version = int(major) * 10000 + int(minnor) * 100
    return version
