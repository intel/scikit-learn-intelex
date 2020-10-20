rem set NO_DIST=1
 
set DAAL4PY_VERSION=%PKG_VERSION%
set MPIROOT=%PREFIX%\Library

IF NOT DEFINED DAALROOT (set DAALROOT=%PREFIX%) 

IF DEFINED DPCPPROOT (
    echo "Sourcing DPCPPROOT"
    call "%DPCPPROOT%\env\vars.bat"
)

set PATH=%PATH%;%PREFIX%\Library\bin\libfabric

%PYTHON% setup.py install
