#!/bin/bash

if [ "$PY3K" == "1" ]; then
    ARGS=""
else
    ARGS="--old-and-unmanageable"
fi

# if dpc++ vars path is specified
if [ ! -z "${DPCPP_VAR}" ]; then
    source ${DPCPP_VAR}
    export CC=dpcpp
fi

# if DAALROOT not exists then provide PREFIX
if [ -z "${DAALROOT}" ]; then
    DAALROOT=${PREFIX}
fi

DAAL4PY_VERSION=$PKG_VERSION TBBROOT=${PREFIX} MPIROOT=${PREFIX} ${PYTHON} setup.py install $ARGS
