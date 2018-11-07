if "%PY3K%"=="1" (
    set ARGS=""
) else (
    set ARGS="--old-and-unmanageable"
)

rem set NO_DIST=1

set DAAL4PY_VERSION=%PKG_VERSION%
set TBBROOT=%PREFIX%
set MPIROOT=%PREFIX%
set DAALROOT=%PREFIX%

%PYTHON% setup.py install %ARGS%
