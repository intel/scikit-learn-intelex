ret_code=0
python -m sklearnex.global patch_sklearn -a svc
ret_code=$(($ret_code + $?))
python -c "from sklearn.svm import SVC; assert SVC.__module__.startswith('daal4py') or SVC.__module__.startswith('sklearnex')"
ret_code=$(($ret_code + $?))
python -c "from sklearn.svm import SVR; assert SVR.__module__.startswith('sklearn')"
ret_code=$(($ret_code + $?))
python -m sklearnex.global unpatch_sklearn
ret_code=$(($ret_code + $?))
python -c "from sklearn.svm import SVC; assert SVC.__module__.startswith('sklearn')"
ret_code=$(($ret_code + $?))
python -c "from sklearn.svm import SVR; assert SVR.__module__.startswith('sklearn')"
ret_code=$(($ret_code + $?))
exit $ret_code
