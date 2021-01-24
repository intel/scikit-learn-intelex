#!/bin/bash
DAAL4PY_ROOT=$1
OUTPUT_ROOT=$2
cd $DAAL4PY_ROOT/.circleci
touch ~/d4p.out ~/skl.out
export DESELECTED_TESTS=$(python deselect_tests.py ../deselected_tests.yaml --reduced)
echo "-m daal4py -m pytest ${DESELECTED_TESTS} -q -ra --disable-warnings --pyargs sklearn"
cd && ((python -m daal4py -m pytest ${DESELECTED_TESTS} -ra --disable-warnings --pyargs sklearn | tee ~/d4p.out) || true)
# extract status strings
export D4P=$(grep -E "=(\s\d*\w*,?)+ in .*\s=" ~/d4p.out)
echo "Summary of patched run: " $D4P
tar cjf $OUTPUT_ROOT ~/d4p.out
python $DAAL4PY_ROOT/.circleci/compare_runs.py
