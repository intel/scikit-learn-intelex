#!/bin/bash

if [ "$PY3K" == "1" ]; then
    ARGS=""
else
    ARGS="--old-and-unmanageable"
fi

export NO_DIST=1

DAAL4PY_VERSION=$PKG_VERSION CNCROOT=${PREFIX} TBBROOT=${PREFIX} DAALROOT=${PREFIX} ${PYTHON} setup.py install $ARGS
