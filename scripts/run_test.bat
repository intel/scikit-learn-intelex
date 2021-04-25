@echo off
rem ============================================================================
rem Copyright 2018-2021 Intel Corporation
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

python -m unittest discover -v -s daal4py\tests -p test*.py
set errorcode=!errorlevel!
if !errorcode! NEQ 0 (
    echo DAAL4PY TEST FAILED
    exit /b 1
)
pytest --pyargs daal4py\sklearn\
set errorcode=!errorlevel!
if !errorcode! NEQ 0 (
    echo DAAL4PY TEST FAILED
    exit /b 1
)

pytest --pyargs sklearnex\tests\
set errorcode=!errorlevel!
if !errorcode! NEQ 0 (
    echo SKLEARNEX TEST FAILED
    exit /b 1
)
echo DAAL4PY TEST PASSED
