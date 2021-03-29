#!/bin/bash
DAAL4PY_ROOT=$1
OUTPUT_ROOT=$2
PACKAGE=$3
OUT_FILE=$4
cd $DAAL4PY_ROOT/.circleci
touch ~/$OUT_FILE.out
export DESELECTED_TESTS=$(python deselect_tests.py ../deselected_tests.yaml --absolute --reduced --public)
echo "-m ${PACKAGE} -m pytest ${DESELECTED_TESTS} -q -ra --disable-warnings --pyargs sklearn"
cd && ((python -m ${PACKAGE} -m pytest ${DESELECTED_TESTS} -ra --disable-warnings --pyargs sklearn | tee ~/${OUT_FILE}.out) || true)
# extract status strings
export D4P=$(grep -E "=(\s\d*\w*,?)+ in .*\s=" ~/${OUT_FILE}.out)
echo "Summary of patched run: " $D4P
tar cjf $OUTPUT_ROOT ~/$OUT_FILE.out
python $DAAL4PY_ROOT/.circleci/compare_runs.py
