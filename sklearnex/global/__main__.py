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

    # Adding custom extend action for support all python versions
    class ExtendAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            items = getattr(namespace, self.dest) or []
            items.extend(values)
            setattr(namespace, self.dest, items)

    parser = argparse.ArgumentParser(
        prog="python -m sklearnex.global",
        description="""
            Patch your all Scikit-learn applications using Intel(R) Extension for
            scikit-learn, optimizing solvers of
            scikit-learn with Intel(R) oneAPI Data Analytics Library.
            """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.register('action', 'extend', ExtendAction)
    parser.add_argument('action', choices=["patch_sklearn", "unpatch_sklearn"],
                        help="Enable or Disable patching")
    parser.add_argument('--no-verbose', '-nv', action='store_true',
                        help="Additional info about patching")
    parser.add_argument('--algorithm', '-a', action='extend', type=str, nargs="+",
                        help="Name of algorithm to global patch")
    args = parser.parse_args()

    if args.action == "patch_sklearn":
        patch_sklearn(name=args.algorithm, verbose=(not args.no_verbose), _global=True)
        return

    if args.action == "unpatch_sklearn":
        unpatch_sklearn(_global=True)
        return


sys.exit(_main())
