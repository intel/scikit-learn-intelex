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

IF DEFINED PKG_VERSION (set DAAL4PY_VERSION=%PKG_VERSION%)

set MPIROOT=%PREFIX%\Library

IF NOT DEFINED DALROOT (set DALROOT=%PREFIX%)

set "BUILD_ARGS="

set PATH=%PATH%;%PREFIX%\Library\bin\libfabric

%PYTHON% setup.py build %BUILD_ARGS%
IF %ERRORLEVEL% neq 0 EXIT /b %ERRORLEVEL%
%PYTHON% setup.py install --single-version-externally-managed --record record.txt
