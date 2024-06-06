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

%PYTHON% -m unittest discover -v -s %1\tests -p test*.py || set exitcode=1

pytest --verbose --pyargs %1\daal4py\sklearn || set exitcode=1
pytest --verbose --pyargs sklearnex || set exitcode=1
pytest -rs --verbose --pyargs %1\onedal --deselect="onedal/common/tests/test_policy.py" || set exitcode=1
python %1\.ci\scripts\test_global_patch.py || set exitcode=1
EXIT /B %exitcode%
