return_code=0

IDP_SKLEARN_VERBOSE=INFO python daal4py/sklearn/test/test_patching/launch_algorithms.py > daal4py/sklearn/test/test_patching/raw_log
return_code=$(($return_code + $?))

python daal4py/sklearn/test/test_patching/results_parser.py
return_code=$(($return_code + $?))

exit $return_code