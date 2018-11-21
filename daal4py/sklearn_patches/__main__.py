#!/usr/bin/env python
#
# Copyright 2016-2018 Intel Corporation.  All Rights Reserved.
#
# The source code contained or described herein and all documents related
# to the source code ("Material") are owned by Intel Corporation or its
# suppliers or licensors.  Title to the Material remains with Intel
# Corporation or its suppliers and licensors.  The Material is protected
# by worldwide copyright laws and treaty provisions.  No part of the
# Material may be used, copied, reproduced, modified, published, uploaded,
# posted, transmitted, distributed, or disclosed in any way without
# Intel's prior express written permission.
#
# No license under any patent, copyright, trade secret or other
# intellectual property right is granted to or conferred upon you by
# disclosure or delivery of the Materials, either expressly, by
# implication, inducement, estoppel or otherwise.  Any license under such
# intellectual property rights must be express and approved by Intel in
# writing.


import sys
from .dispatcher import enable

def _main():
    import argparse

    parser = argparse.ArgumentParser(prog="python -m daal4py.sklearn_patches", description="""
                Run your Python script with Intel(R) Distribution for Python* patches of scikit-learn,
                optimizing solvers of scikit-learn with Intel(R) DAAL.
             """, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-m', action='store_true', dest='module',
                        help="Executes following as a module")
    parser.add_argument('name', help="Script or module name")
    parser.add_argument('args', nargs=argparse.REMAINDER,
                        help="Command line arguments")
    args = parser.parse_args()

    try:
        import sklearn
        enable()
    except ImportError:
        print("Scikit-learn could not be imported. Nothing to patch")

    sys.argv = [args.name] + args.args
    if '_' + args.name in globals():
        return globals()['_' + args.name](*args.args)
    else:
        import runpy
        runf = runpy.run_module if args.module else runpy.run_path
        runf(args.name, run_name='__main__')


sys.exit(_main())
