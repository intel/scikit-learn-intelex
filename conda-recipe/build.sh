#!/bin/bash

if [ "$PY3K" == "1" ]; then
    ARGS="--single-version-externally-managed --record=record.txt"
else
    ARGS="--old-and-unmanageable"
fi

# if dpc++ vars path is specified
if [ ! -z "${DPCPPROOT}" ]; then
    source ${DPCPPROOT}/env/vars.sh
    export CC=dpcpp
    export CXX=dpcpp
    dpcpp --version
fi

# if DAALROOT not exists then provide PREFIX
if [ "${DAALROOT}" != "" ] && [ "${DALROOT}" == "" ] ; then
    export DALROOT="${DAALROOT}"
fi

if [ -z "${DALROOT}" ]; then
    export DALROOT=${PREFIX}
fi

if [ "$(uname)" == "Darwin" ]; then
    # dead_strip_dylibs does not work with DAAL, which is underlinked by design
    export LDFLAGS="${LDFLAGS//-Wl,-dead_strip_dylibs}"
    export LDFLAGS_LD="${LDFLAGS_LD//-dead_strip_dylibs}"
    # some dead_strip_dylibs come from Python's sysconfig. Setting LDSHARED overrides that
    export LDSHARED="-bundle -undefined dynamic_lookup -flto -Wl,-export_dynamic -Wl,-pie -Wl,-headerpad_max_install_names"
fi

export DAAL4PY_VERSION=$PKG_VERSION
export MPIROOT=${PREFIX}
${PYTHON} setup.py install $ARGS
