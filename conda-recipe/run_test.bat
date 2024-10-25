@echo off
rem ============================================================================
rem Copyright 2018 Intel Corporation
rem
rem Licensed under the Apache License, Version 2.0 (the "License");
rem you may not use this file except in compliance with the License.
rem You may obtain a copy of the License at
rem
rem     http://www.apache.org/licenses/LICENSE-2.0
rem
rem Unless required by applicable law or agreed to in writing, software
rem distributed under the License is distributed on an "AS IS" BASIS,
rem WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
rem See the License for the specific language governing permissions and
rem limitations under the License.
rem ============================================================================

rem %1 - scikit-learn-intelex repo root (should end with '\', leave empty if it's %cd% / $PWD)

set exitcode=0

IF NOT DEFINED PYTHON (set "PYTHON=python")

IF DEFINED COVERAGE_RCFILE (set COVARGS=--cov=onedal --cov=sklearnex --cov-config=%COVERAGE_RCFILE% --cov-append --cov-report=) else (set COVARGS=)

%PYTHON% -c "from sklearnex import patch_sklearn; patch_sklearn()" || set exitcode=1

rem Note: execute with argument --json-report as second argument
rem in order to produce a JSON report under folder '.pytest_reports'.
set with_json_report=0
if "%~2"=="--json-report" (
    set with_json_report=1
    mkdir .pytest_reports
    del /q .pytest_reports\*.json
)

if "%with_json_report%"=="1" (
    %PYTHON% -m pytest --verbose -s %1tests --json-report --json-report-file=.pytest_reports\legacy_report.json %COVARGS% || set exitcode=1
    pytest --verbose --pyargs daal4py --json-report --json-report-file=.pytest_reports\daal4py_report.json %COVARGS% || set exitcode=1
    pytest --verbose --pyargs sklearnex --json-report --json-report-file=.pytest_reports\sklearnex_report.json %COVARGS% || set exitcode=1
    pytest --verbose --pyargs onedal --json-report --json-report-file=.pytest_reports\onedal_report.json %COVARGS% || set exitcode=1
    pytest --verbose %1.ci\scripts\test_global_patch.py --json-report --json-report-file=.pytest_reports\global_patching_report.json %COVARGS% || set exitcode=1
    if NOT EXIST .pytest_reports\legacy_report.json (
        echo "Error: JSON report files failed to be produced."
        set exitcode=1
    )
) else (
    %PYTHON% -m pytest --verbose -s %1tests %COVARGS% || set exitcode=1
    pytest --verbose --pyargs daal4py %COVARGS% || set exitcode=1
    pytest --verbose --pyargs sklearnex %COVARGS% || set exitcode=1
    pytest --verbose --pyargs onedal %COVARGS% || set exitcode=1
    pytest --verbose %1.ci\scripts\test_global_patch.py %COVARGS% || set exitcode=1
)

EXIT /B %exitcode%
