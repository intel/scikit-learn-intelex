#!/bin/bash
#===============================================================================
# Copyright 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

DAAL4PY_ROOT=$1
OUTPUT_ROOT=$2
PACKAGE=$3
OUT_FILE=$4
cd $DAAL4PY_ROOT/.circleci
touch ~/$OUT_FILE.out
export DESELECTED_TESTS=$(python deselect_tests.py ../deselected_tests.yaml --absolute --reduced --public)
echo "-m ${PACKAGE} -m pytest ${DESELECTED_TESTS} -q -ra --disable-warnings --pyargs sklearn"
cd && ((python -m ${PACKAGE} -m pytest ${DESELECTED_TESTS} -ra --disable-warnings --verbose --log-cli-level=DEBUG --pyargs sklearn | tee ~/${OUT_FILE}.out) || true)
# extract status strings
export D4P=$(grep -E "=(\s\d*\w*,?)+ in .*\s=" ~/${OUT_FILE}.out)
echo "Summary of patched run: " $D4P
tar cjf $OUTPUT_ROOT ~/$OUT_FILE.out
python $DAAL4PY_ROOT/.circleci/compare_runs.py
