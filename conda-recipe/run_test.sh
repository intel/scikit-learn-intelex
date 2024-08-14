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

daal4py_dir="$( cd "$( dirname "$( dirname "${BASH_SOURCE[0]}" )")" && pwd )"

return_code=0

if [ -z "${PYTHON}" ]; then
    export PYTHON=python
fi

${PYTHON} -c "from sklearnex import patch_sklearn; patch_sklearn()"
return_code=$(($return_code + $?))

echo "NO_DIST=$NO_DIST"
if [[ ! $NO_DIST ]]; then
    echo "MPI unittest discover testing ..."
    mpirun --version
    mpirun -n 4 python -m unittest discover -v -s ${daal4py_dir}/tests -p test*spmd*.py
    return_code=$(($return_code + $?))
fi

${PYTHON} -m unittest discover -v -s ${daal4py_dir}/tests -p test*.py
return_code=$(($return_code + $?))

pytest --verbose --pyargs daal4py
return_code=$(($return_code + $?))

pytest --verbose --pyargs sklearnex
return_code=$(($return_code + $?))

pytest --verbose --pyargs onedal
return_code=$(($return_code + $?))

${PYTHON} ${daal4py_dir}/.ci/scripts/test_global_patch.py
return_code=$(($return_code + $?))

exit $return_code
