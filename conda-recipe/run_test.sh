#!/bin/bash

#*******************************************************************************
# Copyright 2014-2021 Intel Corporation
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

ok=0

echo "Running import daal4py"
python -c "import daal4py"
kek=$?
echo "Exit code = $kek"
ok=$(($ok + $kek))

if ! $NO_DIST; then
    echo "Running mpirun"
    mpirun -n 4 python -m unittest discover -v -s tests -p spmd*.py
    kek=$?
    echo "Exit code = $kek"
    ok=$(($ok + $kek))
fi

echo "Running pytest"
pytest --pyargs daal4py/sklearn/
kek=$?
echo "Exit code = $kek"
ok=$(($ok + $kek))

echo "Running unittest"
python -m unittest discover -v -s tests -p test*.py
kek=$?
echo "Exit code = $kek"
ok=$(($ok + $kek))

exit $ok
