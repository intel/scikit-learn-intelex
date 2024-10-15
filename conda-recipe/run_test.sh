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

sklex_root="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
count=3
while [[ count -ne 0 ]]; do
    if [[ -d $sklex_root/.ci/ && -d $sklex_root/examples/ && -d $sklex_root/tests/ ]]; then
        break
    fi
    sklex_root="$( dirname "${sklex_root}" )"
    count=$(($count - 1))
done

if [[ count -eq 0 ]]; then
    echo "run_test.sh did not find the required testing directories"
    exit 1
fi

return_code=0

if [ -z "${PYTHON}" ]; then
    export PYTHON=python
fi

${PYTHON} -c "from sklearnex import patch_sklearn; patch_sklearn()"
return_code=$(($return_code + $?))

echo "NO_DIST=$NO_DIST"
if [[ ! $NO_DIST ]]; then
    echo "MPI pytest run"
    mpirun --version
    mpirun -n 4 pytest --verbose -s ${sklex_root}/tests/test*spmd*.py
    return_code=$(($return_code + $?))
fi

pytest --verbose -s ${sklex_root}/tests
return_code=$(($return_code + $?))

pytest --verbose --pyargs daal4py
return_code=$(($return_code + $?))

pytest --verbose --pyargs sklearnex
return_code=$(($return_code + $?))

pytest --verbose --pyargs onedal
return_code=$(($return_code + $?))

pytest --verbose -s ${sklex_root}/.ci/scripts/test_global_patch.py
return_code=$(($return_code + $?))

exit $return_code
