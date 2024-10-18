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

echo "Start testing ..."
return_code=0

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

python -c "import daal4py"
return_code=$(($return_code + $?))

echo "Pytest run of legacy unittest ..."
echo ${daal4py_dir}
pytest --verbose -s ${daal4py_dir}/tests $@ $(json_report_name legacy)
return_code=$(($return_code + $?))

echo "NO_DIST=$NO_DIST"
if [[ ! $NO_DIST ]]; then
    echo "MPI pytest run of legacy unittest ..."
    mpirun --version
    mpirun -n 4 pytest --verbose -s ${daal4py_dir}/tests/test*spmd*.py $@ $(json_report_name mpi_legacy)
    return_code=$(($return_code + $?))
fi

echo "Pytest of daal4py running ..."
pytest --verbose --pyargs ${daal4py_dir}/daal4py/sklearn $@ $(json_report_name daal4py)
return_code=$(($return_code + $?))

echo "Pytest of sklearnex running ..."
pytest --verbose --pyargs sklearnex $@ $(json_report_name sklearnex)
return_code=$(($return_code + $?))

echo "Pytest of onedal running ..."
pytest --verbose --pyargs ${daal4py_dir}/onedal $@ $(json_report_name onedal)
return_code=$(($return_code + $?))

echo "Global patching test running ..."
pytest --verbose -s ${daal4py_dir}/.ci/scripts/test_global_patch.py $@ $(json_report_name global_patching)
return_code=$(($return_code + $?))

if [[ "${with_json_report}" == "1" ]]; then
    if [[ ! -f ".pytest_reports/legacy_report.json" ]]; then
        echo "Error: JSON report files failed to be produced."
        return_code=1
    fi
fi

exit $return_code
