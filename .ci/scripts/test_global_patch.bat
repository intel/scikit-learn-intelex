set /a RET_CODE=0
python -m sklearnex.global patch_sklearn -a svc
set /a RET_CODE=%RET_CODE% + %ERRORLEVEL%
python -c "from sklearn.svm import SVC; assert SVC.__module__.startswith('daal4py') or SVC.__module__.startswith('sklearnex')"
set /a RET_CODE=%RET_CODE% + %ERRORLEVEL%
python -c "from sklearn.svm import SVR; assert SVR.__module__.startswith('sklearn')"
set /a RET_CODE=%RET_CODE% + %ERRORLEVEL%
python -m sklearnex.global unpatch_sklearn
set /a RET_CODE=%RET_CODE% + %ERRORLEVEL%
python -c "from sklearn.svm import SVC; assert not (SVC.__module__.startswith('daal4py') or SVC.__module__.startswith('sklearnex'))"
set /a RET_CODE=%RET_CODE% + %ERRORLEVEL%
python -c "from sklearn.svm import SVR; assert SVR.__module__.startswith('sklearn')"
set /a RET_CODE=%RET_CODE% + %ERRORLEVEL%
IF %ERRORLEVEL% neq 0 EXIT /b %ERRORLEVEL%
