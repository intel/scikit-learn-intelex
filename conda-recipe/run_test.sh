#*******************************************************************************
# Copyright 2014-2020 Intel Corporation
# All Rights Reserved.
#
# This software is licensed under the Apache License, Version 2.0 (the
# "License"), the following terms apply:
#
# You may not use this file except in compliance with the License.  You may
# obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# See the License for the specific language governing permissions and
# limitations under the License.
#*******************************************************************************

#!/bin/bash

# if dpc++ vars path is specified
if [ ! -z "${DPCPPROOT}" ]; then
    source ${DPCPPROOT}/env/vars.sh
fi

# if DAALROOT is specified
if [ ! -z "${DAALROOT}" ]; then
    conda remove daal --force -y
    source ${DAALROOT}/env/vars.sh
fi

# if TBBROOT is specified
if [ ! -z "${TBBROOT}" ]; then
    conda remove tbb --force -y
    source ${TBBROOT}/env/vars.sh
fi
