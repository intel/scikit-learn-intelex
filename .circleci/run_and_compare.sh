#!/bin/bash
DAAL4PY_ROOT=$1
OUTPUT_ROOT=$2
cd $DAAL4PY_ROOT/.circleci
touch ~/d4p.out ~/skl.out
export DESELECTED_TESTS=`python deselect_tests.py ../deselected_tests.yaml --absolute`
echo "-m daal4py -m pytest ${DESELECTED_TESTS} -q -ra --disable-warnings --pyargs sklearn"
# It is important for pytest to work in a separate test folder to not discover tests in ~/miniconda folder
cd && ((python -m daal4py -m pytest ${DESELECTED_TESTS} -ra --disable-warnings --pyargs sklearn | tee ~/d4p.out) || true)
cd && ((python -m pytest ${DESELECTED_TESTS} -ra --disable-warnings --pyargs sklearn | tee ~/skl.out) || true)
# extract status strings
export D4P=`grep -E "=(\s\d*\w*,?)+ in .*\s=" ~/d4p.out`
export SKL=`grep -E "=(\s\d*\w*,?)+ in .*\s=" ~/skl.out`
tar cjf $1 ~/d4p.out ~/skl.out
echo "Summary of patched run: " $D4P
echo "Summary of unpatched run: " $SKL
python $DAAL4PY_ROOT/.circleci/compare_runs.py
