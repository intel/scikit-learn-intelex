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

# Note: execute with argument --json-report in order to produce
# a JSON report under folder '.pytest_reports'. Other arguments
# will also be forwarded to pytest.
with_json_report=0
if [[ "$*" == *"--json-report"* ]]; then
    echo "Will produce JSON report of tests"
    with_json_report=1
    mkdir -p .pytest_reports
    if [[ ! -z "$(ls .pytest_reports)" ]]; then
        rm .pytest_reports/*.json
    fi
fi
function json_report_name {
    if [[ "${with_json_report}" == "1" ]]; then
        printf -- "--json-report-file=.pytest_reports/$1_report.json"
    fi
}

${PYTHON} -c "from sklearnex import patch_sklearn; patch_sklearn()"
return_code=$(($return_code + $?))

pytest --verbose -s "${sklex_root}/tests" $@ $(json_report_name legacy)
return_code=$(($return_code + $?))

pytest --verbose --pyargs daal4py $@ $(json_report_name daal4py)
return_code=$(($return_code + $?))

pytest --verbose --pyargs sklearnex $@ $(json_report_name sklearnex)
return_code=$(($return_code + $?))

pytest --verbose --pyargs onedal $@ $(json_report_name onedal)
return_code=$(($return_code + $?))

pytest --verbose -s "${sklex_root}/.ci/scripts/test_global_patch.py" $@ $(json_report_name global_patching)
return_code=$(($return_code + $?))

echo "NO_DIST=$NO_DIST"
if [[ ! $NO_DIST ]]; then
    mpirun --version
    mpirun -n 4 python "${sklex_root}/conda-recipe/helper_mpi_tests.py" \
        "pytest --verbose -s \"${sklex_root}/tests/test_daal4py_spmd_examples.py\"" \
        "pytest --verbose -s \"${sklex_root}/tests/test_daal4py_spmd_examples.py\" $@ $(json_report_name mpi_legacy)"
    return_code=$(($return_code + $?))
fi

exit $return_code
