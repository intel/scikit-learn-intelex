#!/bin/bash

if [ -n "$PY_VER" ] && [ "${PY_VER:0:1}" -lt "3" ]; then
    ARG="--old-and-unmanageable"
else
    ARG=""
fi
DAAL4PY_VERSION=$PKG_VERSION CNCROOT=${PREFIX} TBBROOT=${PREFIX} DAALROOT=${PREFIX} ${PYTHON} setup.py install $ARG
