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

daal4py_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
count=3
while [[ count -ne 0 ]]; do
    if [[ -d $daal4py_dir/daal4py/ && -d $daal4py_dir/tests/ && -d $daal4py_dir/examples/ ]]; then
        break
    fi
    daal4py_dir="$( dirname "${daal4py_dir}" )"
    count=$(($count - 1))
done

echo "daal4py_dir=$daal4py_dir"
if [[ count -eq 0 ]]; then
    echo "run_test.sh must be in daal4py repository"
    exit 1
fi

echo "Start testing ..."
return_code=0

python -c "import daal4py"
return_code=$(($return_code + $?))

echo "NO_DIST=$NO_DIST"
if [[ ! $NO_DIST ]]; then
    echo "MPI unittest discover testing ..."
    mpirun --version
    mpirun -n 4 python -m unittest discover -v -s ${daal4py_dir}/tests -p spmd*.py
    return_code=$(($return_code + $?))
fi

echo "Unittest discover testing ..."
python -m unittest discover -v -s ${daal4py_dir}/tests -p test*.py
return_code=$(($return_code + $?))

echo "Pytest running ..."
pytest --pyargs ${daal4py_dir}/daal4py/sklearn/
return_code=$(($return_code + $?))

exit $return_code
