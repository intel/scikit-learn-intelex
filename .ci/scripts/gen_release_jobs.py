#! /usr/bin/env python
# ==============================================================================
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
# ==============================================================================

import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--channels", nargs="+", default=["pypi"])
args = parser.parse_args()

CHANNELS = args.channels
PYTHON_VERSIONS = ["3.9", "3.10", "3.11"]
SYSTEMS = ["ubuntu-22.04", "windows-2022"]
ACTIVATE = {
    "ubuntu-latest": "conda activate",
    "windows-latest": "call activate",
}

print(CHANNELS)

res_enum = {}
for channel in CHANNELS:
    for python_version in PYTHON_VERSIONS:
        for os in SYSTEMS:
            res_key = channel + " - " + "python" + python_version + " - " + os
            res_enum[res_key] = {}
            res_enum[res_key]["python.version"] = python_version
            res_enum[res_key]["imageName"] = os
            res_enum[res_key]["conda.activate"] = ACTIVATE[os]
            res_enum[res_key]["conda.channel"] = channel

sys.stderr.write("##vso[task.setVariable variable=legs;isOutput=true]{}".format(res_enum))
