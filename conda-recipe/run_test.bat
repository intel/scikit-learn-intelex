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

rem %1% - scikit-learn-intelex repo root (leave empty if it's %cd% / $PWD)

set exitcode=0

IF NOT DEFINED PYTHON (set "PYTHON=python")

%PYTHON% -c "from sklearnex import patch_sklearn; patch_sklearn()" || set exitcode=1

%PYTHON% -m pytest --verbose -s %1%tests || set exitcode=1

pytest --verbose --pyargs daal4py || set exitcode=1
pytest --verbose --pyargs sklearnex || set exitcode=1
pytest --verbose --pyargs onedal || set exitcode=1
pytest --verbose %1%.ci\scripts\test_global_patch.py || set exitcode=1
EXIT /B %exitcode%
