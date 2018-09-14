#!/bin/bash

if [ "$PY3K" == "1" ]; then
    ARGS=""
else
    ARGS="--old-and-unmanageable"
fi

DAAL4PY_VERSION=$PKG_VERSION CNCROOT=${PREFIX} TBBROOT=${PREFIX} DAALROOT=${PREFIX} ${PYTHON} setup.py install $ARGS
