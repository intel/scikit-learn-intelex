#!/bin/bash

if [ "$PY3K" == "1" ]; then
    ARGS=""
else
    ARGS="--old-and-unmanageable"
fi

#export NO_DIST=1

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
export DAALROOT=${PREFIX}
${PYTHON} setup.py install $ARGS
