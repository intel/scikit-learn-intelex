#!/bin/bash
#===============================================================================
# Copyright 2018 Intel Corporation
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

if [ "$PY3K" == "1" ]; then
    ARGS=" --single-version-externally-managed --record=record.txt"
else
    ARGS="--old-and-unmanageable"
fi

if [ -z "${DALROOT}" ]; then
    export DALROOT=${PREFIX}
fi

if [ "$(uname)" == "Darwin" ]; then
    export CC=gcc
    export CXX=g++
fi

if [ ! -z "${PKG_VERSION}" ]; then
    export DAAL4PY_VERSION=$PKG_VERSION
fi
export MPIROOT=${PREFIX}
${PYTHON} setup.py install $ARGS
