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

check_status () {
    if [ $? -ne 0 ]; then
        ok=1
    fi
}

python -c "import daal4py"
check_status

mpirun -n 4 python -m unittest discover -v -s tests -p spmd*.py
check_status

pytest --pyargs daal4py/sklearn/neighbors/tests
check_status

python tests/run_tests.py
check_status

exit $ok
