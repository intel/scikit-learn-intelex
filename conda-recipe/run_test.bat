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

rem %1 - scikit-learn-intelex repo root

set MPIROOT=%PREFIX%\Library
set exitcode=0

IF DEFINED DPCPPROOT (
    echo "Sourcing DPCPPROOT"
    call "%DPCPPROOT%\env\vars.bat" || set exitcode=1
    set "CC=dpcpp"
    set "CXX=dpcpp"
    dpcpp --version
)

IF DEFINED DALROOT (
    echo "Sourcing DALROOT"
    call "%DALROOT%\env\vars.bat" || set exitcode=1
    echo "Finish sourcing DALROOT"
)

IF DEFINED TBBROOT (
    echo "Sourcing TBBROOT"
    call "%TBBROOT%\env\vars.bat" || set exitcode=1
)

rem Note: execute with argument --json-report in order to produce
rem a JSON report under folder '.pytest_reports'. Other arguments
rem will also be forwarded to pytest, but '--json-report' needs to
rem be the first one.
if "%~1"=="--json-report" (
    mkdir .pytest_reports
    del /q .pytest_reports\*.json
)

%PYTHON% -m pytest --verbose -s %1\tests %* --json-report-file=.pytest_reports\legacy_report.json || set exitcode=1

pytest --verbose --pyargs %1\daal4py\sklearn %* --json-report-file=.pytest_reports\daal4py_report.json || set exitcode=1
pytest --verbose --pyargs sklearnex %* --json-report-file=.pytest_reports\sklearnex_report.json || set exitcode=1
pytest --verbose --pyargs %1\onedal --deselect="onedal/common/tests/test_policy.py" %* --json-report-file=.pytest_reports\onedal_report.json || set exitcode=1
pytest --verbose %1\.ci\scripts\test_global_patch.py %* --json-report-file=.pytest_reports\global_patching_report.json || set exitcode=1
EXIT /B %exitcode%
