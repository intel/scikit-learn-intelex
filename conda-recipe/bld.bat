if "%PY3K%"=="1" (
    set ARGS=""
) else {
    set ARGS="--old-and-unmanageable"
}

set DAAL4PY_VERSION=%PKG_VERSION%
set CNCROOT=%PREFIX%
set TBBROOT=%PREFIX%
set DAALROOT=%PREFIX%

%PYTHON% setup.py install %ARGS%
