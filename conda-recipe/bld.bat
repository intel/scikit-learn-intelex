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

IF NOT DEFINED PYTHON (set PYTHON="python")
IF DEFINED PKG_VERSION (set SKLEARNEX_VERSION=%PKG_VERSION%)
IF NOT DEFINED DALROOT (set DALROOT=%PREFIX%)
IF NOT DEFINED MPIROOT (set MPIROOT=%PREFIX%\Library)

echo CC=%CC%
echo CXX=%CXX%
echo CMAKE_C_COMPILER=%CMAKE_C_COMPILER%
echo CMAKE_CXX_COMPILER=%CMAKE_CXX_COMPILER%
echo CC_FOR_BUILD=%CC_FOR_BUILD%
echo CXX_FOR_BUILD=%CXX_FOR_BUILD%

rem reset preferred compilers to avoid usage of icx/icpx by default in all cases
set CC=cl.exe
set CXX=cl.exe

rem source compiler if DPCPPROOT is set outside of conda-build
IF DEFINED DPCPPROOT (
    echo "Sourcing DPCPPROOT"
    call "%DPCPPROOT%\env\vars.bat"
)

%PYTHON% setup.py install --single-version-externally-managed --record record.txt
