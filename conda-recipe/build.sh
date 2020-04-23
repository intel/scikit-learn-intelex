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
    export CXX=dpcpp
fi

# if DAALROOT not exists then provide PREFIX
if [ -z "${DAALROOT}" ]; then
    export DAALROOT=${PREFIX}
fi

if [ `uname` == Darwin ]; then
    # dead_strip_dylibs does not work with DAAL, which is underlinked by design
    export LDFLAGS="${LDFLAGS//-Wl,-dead_strip_dylibs}"
    export LDFLAGS_LD="${LDFLAGS_LD//-dead_strip_dylibs}"
    # some dead_strip_dylibs come from Python's sysconfig. Setting LDSHARED overrides that
    export LDSHARED="-bundle -undefined dynamic_lookup -flto -Wl,-export_dynamic -Wl,-pie -Wl,-headerpad_max_install_names"
fi

export DAAL4PY_VERSION=$PKG_VERSION
export TBBROOT=${PREFIX}
export MPIROOT=${PREFIX}
${PYTHON} setup.py install $ARGS
