#!/bin/bash

# if dpc++ vars path is specified
if [ ! -z "${DPCPP_VAR}" ]; then
    source ${DPCPP_VAR}
fi

# if DAALROOT is specified
if [ ! -z "${DAALROOT}" ]; then
    conda remove daal --force -y
    source ${DAALROOT}/env/vars.sh
fi
