#! /usr/bin/env python
# ===============================================================================
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
# ===============================================================================

import re
from os.path import join as jp
from os.path import isfile


def find_defines(defines: list, file_obj):
    defines_dict = {define: "" for define in defines}
    for elem in file_obj:
        for define in defines:
            if f"#define {define}" in elem:
                match = re.match(rf"#define {define} (\d+)", elem)
                if match:
                    defines_dict[define] = match.group(1)
    return defines_dict


def get_onedal_version(dal_root, version_type="release"):
    """Parse oneDAL version strings"""

    if version_type not in ["release", "binary"]:
        raise ValueError(f'Incorrect version type "{version_type}"')

    header_version = jp(dal_root, "include", "dal", "services", "library_version_info.h")
    if not isfile(header_version):
        # pre-2024.0 release header path
        header_version = jp(dal_root, "include", "services", "library_version_info.h")
    version = ""

    with open(header_version, "r") as header:
        if version_type == "release":
            version = find_defines(
                ["__INTEL_DAAL__", "__INTEL_DAAL_MINOR__", "__INTEL_DAAL_UPDATE__"],
                header,
            )
            version = (
                int(version["__INTEL_DAAL__"]) * 10000
                + int(version["__INTEL_DAAL_MINOR__"]) * 100
                + int(version["__INTEL_DAAL_UPDATE__"])
            )
        elif version_type == "binary":
            version = find_defines(
                ["__INTEL_DAAL_MAJOR_BINARY__", "__INTEL_DAAL_MINOR_BINARY__"], header
            )
            version = int(version["__INTEL_DAAL_MAJOR_BINARY__"]), int(
                version["__INTEL_DAAL_MINOR_BINARY__"]
            )
    return version
