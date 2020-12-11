rem set NO_DIST=1

set DAAL4PY_VERSION=%PKG_VERSION%
set MPIROOT=%PREFIX%\Library

IF DEFINED DAALROOT (set DALROOT=%DAALROOT%) 

IF NOT DEFINED DALROOT (set DALROOT=%PREFIX%) 

IF DEFINED DPCPPROOT (
    echo "Sourcing DPCPPROOT"
    call "%DPCPPROOT%\env\vars.bat"
    set "CC=dpcpp"
    set "CXX=dpcpp"
    dpcpp --version
)

set PATH=%PATH%;%PREFIX%\Library\bin\libfabric

SETLOCAL
set ARGS="--single-version-externally-managed --record=record.txt"
%PYTHON% setup.py install %ARGS%
ENDLOCAL
