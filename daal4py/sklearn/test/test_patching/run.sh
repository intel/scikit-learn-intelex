#./daal4py/sklearn/test/test_patching/run.sh
IDP_SKLEARN_VERBOSE=INFO python daal4py/sklearn/test/test_patching/launch_algorithms.py > daal4py/sklearn/test/test_patching/raw_log
python daal4py/sklearn/test/test_patching/results_parser.py > daal4py/sklearn/test/test_patching/final_results