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
bin_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
daal4py_dir="$( dirname "${bin_dir}" )"

python -c "import daal4py"
ok=$(($ok + $?))

if [[ ! $NO_DIST ]]; then
    mpirun -n 4 python -m unittest discover -v -s tests -p ${daal4py_dir}/spmd*.py
    ok=$(($ok + $?))
fi

python -m unittest discover -v -s tests -p ${daal4py_dir}/test*.py
ok=$(($ok + $?))

pytest --pyargs ${daal4py_dir}/daal4py/sklearn/
ok=$(($ok + $?))

exit $ok
