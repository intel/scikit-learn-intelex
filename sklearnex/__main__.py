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

import sys
from sklearnex import patch_sklearn
from sklearnex import unpatch_sklearn


def _main():
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m sklearnex",
        description="""
            Run your Python script with Intel(R) Extension for
            scikit-learn, optimizing solvers of
            scikit-learn with Intel(R) oneAPI Data Analytics Library.
            """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-m', action='store_true', dest='module',
                        help="Executes following as a module")
    parser.add_argument('name', help="Script or module name. Type 'patch' or"
                        " 'unpatch' to, patch/unpatch all scikit-learn applications"
                        " (in this case other arguments must be empty)")
    parser.add_argument('--verbose', '-v', action='store_true',
                        help="Additional info about patching"
                        " (only for global patching)")
    parser.add_argument('--algorithm', '-a', action='append',
                        help="Name of algorithm to global patch")
    parser.add_argument('args', nargs=argparse.REMAINDER,
                        help="Command line arguments")
    args = parser.parse_args()

    patch_sklearn_global_ready = \
        not args.module and not args.args and args.name == "global_patch"
    unpatch_sklearn_global_ready = \
        not args.module and not args.args and args.name == "global_unpatch"

    if patch_sklearn_global_ready:
        patch_sklearn(name=args.algorithm, verbose=not args.verbose, _global=True)
        return

    if unpatch_sklearn_global_ready:
        unpatch_sklearn(name=args.algorithm, _global=True)
        return

    if args.verbose or not args.algorithm:
        raise RuntimeError("-v, --verbose and -a, --algorithm options can be using"
                        " only for global pathcing. For example:"
                        " python -m sklearnex -v -a algo_1 -a algo_2 global_patch")
    try:
        import sklearn
        patch_sklearn()
    except ImportError:
        print("Scikit-learn could not be imported. Nothing to patch")

    sys.argv = [args.name] + args.args
    if '_' + args.name in globals():
        return globals()['_' + args.name](*args.args)
    import runpy
    runf = runpy.run_module if args.module else runpy.run_path
    runf(args.name, run_name='__main__')


sys.exit(_main())
