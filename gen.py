#!/usr/bin/env python
#===============================================================================
# Copyright 2020-2021 Intel Corporation
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

from generator.gen_daal4py import gen_daal4py

if __name__ == "__main__":
    import argparse
    global no_warn

    description = """A tool to create cython interface files for HLAPI of oneDAL (aka daal4py).
    Extracting necessary data and creating internal data structures.
    See parse.py for details about C++ parsing.
    See wrappers.py for necessary configuration that can not be auto-extracted.
    See wrapper_gen.py for code generation (cython and C++).
    """

    argParser = argparse.ArgumentParser(
        prog="gen_daal4py.py",
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argParser.add_argument('--daalroot', required=True,
                           help="DAAL root directory (reads include dir in there)")
    argParser.add_argument('--outdir', default='build',
                           help="Output directory to store wrapper files to")
    argParser.add_argument('--wall', default=False,
                           action='store_true', help="Emit all warnings")

    args = argParser.parse_args()
    gen_daal4py(args.daalroot, args.outdir, "v_develop", args.wall)
