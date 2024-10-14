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

daal4py_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
count=3
while [[ count -ne 0 ]]; do
    if [[ -d $daal4py_dir/daal4py/ && -d $daal4py_dir/tests/ && -d $daal4py_dir/examples/daal4py ]]; then
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

COV_ARGS=(--no-cov)
if [[ "$COVERAGE" == "true" ]]; then
COV_ARGS=(--cov-config "$daal4py_dir/.coveragerc" --cov-report term)
fi

echo "Start testing ..."
return_code=0

python -c "import daal4py"
return_code=$(($return_code + $?))

echo "Pytest run of legacy unittest ..."
echo ${daal4py_dir}
pytest --verbose -s ${daal4py_dir}/tests
return_code=$(($return_code + $?))

echo "NO_DIST=$NO_DIST"
if [[ ! $NO_DIST ]]; then
    echo "MPI pytest run of legacy unittest ..."
    mpirun --version
    mpirun -n 4 pytest --verbose -s ${daal4py_dir}/tests/test*spmd*.py
    return_code=$(($return_code + $?))
fi

echo "Pytest of daal4py running ..."
pytest --verbose ${daal4py_dir}/daal4py/sklearn "${COV_ARGS[@]}"
return_code=$(($return_code + $?))

echo "Pytest of sklearnex running ..."
pytest --verbose --pyargs sklearnex "${COV_ARGS[@]}"
return_code=$(($return_code + $?))

echo "Pytest of onedal running ..."
pytest --verbose --pyargs onedal 
return_code=$(($return_code + $?))

echo "Global patching test running ..."
pytest --verbose -s ${daal4py_dir}/.ci/scripts/test_global_patch.py
return_code=$(($return_code + $?))

exit $return_code
