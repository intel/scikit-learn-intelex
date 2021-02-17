rem set NO_DIST=1
 
set DAAL4PY_VERSION=%PKG_VERSION%
set MPIROOT=%PREFIX%\Library

IF DEFINED DAALROOT (set DALROOT=%DAALROOT%) 

IF NOT DEFINED DALROOT (set DALROOT=%PREFIX%) 

set "BUILD_ARGS="

IF DEFINED DPCPPROOT (
    echo "Sourcing DPCPPROOT"
    call "%DPCPPROOT%\env\vars.bat"
    set "CC=clang-cl"
    set "CXX=clang-cl"
    clang-cl --version
    SET "BUILD_ARGS=--compiler clang-cl"
)

set PATH=%PATH%;%PREFIX%\Library\bin\libfabric

%PYTHON% setup.py build %BUILD_ARGS%
IF %ERRORLEVEL% neq 0 EXIT /b %ERRORLEVEL%
%PYTHON% setup.py install --single-version-externally-managed --record record.txt
