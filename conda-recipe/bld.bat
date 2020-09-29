rem set NO_DIST=1

set PATH=%PATH%;%PREFIX%\Library\bin\libfabric

rem if dpc++ vars path is specified
IF DEFINED DPCPPROOT (
    echo "=============DPCPP==============="
    call "%DPCPPROOT%\env\vars.bat"
    set "CC=dpcpp.exe"
    set "CXX=dpcpp.exe"
)

rem if DAALROOT not exists then provide PREFIX
rem otherwise use library directly from the path provided

IF NOT DEFINED DAALROOT ( 
    set DAALROOT=%PREFIX% 
) ELSE (
    call conda remove daal daal-include --force -y
)

set DAALROOT=%PREFIX% 
set DAAL4PY_VERSION=%PKG_VERSION%
set MPIROOT=%PREFIX%\Library

%PYTHON% setup.py install
