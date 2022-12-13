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

IF DEFINED DPCPPROOT (
    echo "Sourcing DPCPPROOT"
    call "%DPCPPROOT%\env\vars.bat"
    set "CC=dpcpp"
    set "CXX=dpcpp"
    dpcpp --version
)

IF DEFINED DAALROOT (set DALROOT=%DAALROOT%)

IF DEFINED DALROOT (
    echo "Sourcing DALROOT"
    call "%DALROOT%\env\vars.bat"
    echo "Finish sourcing DALROOT"
)

IF DEFINED TBBROOT (
    echo "Sourcing TBBROOT"
    call "%TBBROOT%\env\vars.bat"
)

%PYTHON% -m unittest discover -v -s %1\tests -p test*.py

pytest --verbose --pyargs %1\daal4py\sklearn --deselect="daal4py/sklearn/ensemble/tests/test_decision_forest.py::test_classifier_big_estimators_iris[8000]" --deselect="daal4py/sklearn/ensemble/tests/test_decision_forest.py::test_mse_regressor_big_estimators_iris[8000]"
pytest --verbose --pyargs %1\sklearnex
