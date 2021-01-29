#!/bin/bash

#*******************************************************************************
# Copyright 2014-2021 Intel Corporation
# All Rights Reserved.
#
# This software is licensed under the Apache License, Version 2.0 (the
# "License"), the following terms apply:
#
# You may not use this file except in compliance with the License.  You may
# obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# See the License for the specific language governing permissions and
# limitations under the License.
#*******************************************************************************

ok=0

if ! python -c "import daal4py"; then
    ok=1
fi
if ! mpirun -n 4 python -m unittest discover -v -s tests -p spmd*.py; then
    ok=1
fi
if ! pytest --pyargs daal4py/sklearn/neighbors/tests; then
    ok=1
fi
if ! python tests/run_tests.py; then
    ok=1
fi
exit $ok
