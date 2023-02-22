from cmath import sqrt
import numpy as np
import sys
import os

root_folder = os.path.abspath(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
    )
)
sys.path.append(root_folder)
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, log_loss
from sklearn.ensemble import RandomForestClassifier as ScikitRandomForestClassifier
from daal4py.sklearn.ensemble import RandomForestClassifier as DaalRandomForestClassifier
from sklearn.ensemble import RandomForestRegressor as ScikitRandomForestRegressor
from daal4py.sklearn.ensemble import RandomForestRegressor as DaalRandomForestRegressor




from daal4py.sklearn._utils import daal_check_version
import numpy
IRIS = fetch_openml(data_id=40922)
import time
RNG = np.random.RandomState(0)
WARM_START_STATUS=True
def test_warmstart_sklearn_sklearnex_classifier():
   
    
    daal4py_model = DaalRandomForestClassifier(
        n_estimators=10,
        max_depth=None,
        random_state=777,
        warm_start=WARM_START_STATUS,
    )
    scikit_model = ScikitRandomForestClassifier(
        n_estimators=10,
        max_depth=None,
        random_state=777,
        warm_start=WARM_START_STATUS,
    )
    # training
    
    curr_time = time.time()
    
    daal4py_model.fit(IRIS.data[:10000], IRIS.target[:10000])
    daal4py_model.n_estimators +=50
    daal4py_model.fit(IRIS.data[10001:20000], IRIS.target[10001:20000])
    daal4py_model.n_estimators +=50
    daal4py_model.fit(IRIS.data[20001:30000], IRIS.target[20001:30000])
    daal4py_model.n_estimators +=50
    daal4py_model.fit(IRIS.data[40001:50000], IRIS.target[40001:50000])
    daal4py_model.n_estimators +=50
    daal4py_model.fit(IRIS.data[50001:60000], IRIS.target[50001:60000])
    daal4py_model.n_estimators +=50
    daal4py_model.fit(IRIS.data[60001:70000], IRIS.target[60001:70000])
    daal4py_model.n_estimators +=50
    daal4py_model.fit(IRIS.data[70001:80000], IRIS.target[70001:80000])
    
    print(time.time()- curr_time)

    # training
    
    curr_time = time.time()
    
    scikit_model.fit(IRIS.data[:10000], IRIS.target[:10000])
    scikit_model.n_estimators +=50
    scikit_model.fit(IRIS.data[10001:20000], IRIS.target[10001:20000])
    scikit_model.n_estimators +=50
    scikit_model.fit(IRIS.data[20001:30000], IRIS.target[20001:30000])
    scikit_model.n_estimators +=50
    scikit_model.fit(IRIS.data[40001:50000], IRIS.target[40001:50000])
    scikit_model.n_estimators +=50
    scikit_model.fit(IRIS.data[50001:60000], IRIS.target[50001:60000])
    scikit_model.n_estimators +=50
    scikit_model.fit(IRIS.data[60001:70000], IRIS.target[60001:70000])
    scikit_model.n_estimators +=50
    scikit_model.fit(IRIS.data[70001:80000], IRIS.target[70001:80000])
    
    print(time.time()- curr_time)

    scikit_predict = scikit_model.predict(IRIS.data[80001:])
    daal4py_predict = daal4py_model.predict(IRIS.data[80001:])
    #accuracy
    scikit_accuracy = accuracy_score(scikit_predict, IRIS.target[80001:])
    daal4py_accuracy = accuracy_score(daal4py_predict, IRIS.target[80001:])
    ratio = daal4py_accuracy / scikit_accuracy  
    print("scikit_accuracy :"+str(scikit_accuracy))
    print("daal4py_accuracy :"+str(daal4py_accuracy))
    print("accuracy ratio patched:orignal :"+str(ratio))
   

    # predict_proba
    scikit_predict_proba = scikit_model.predict_proba(IRIS.data[80001:])
    daal4py_predict_proba = daal4py_model.predict_proba(IRIS.data[80001:])
    #log_loss
    scikit_log_loss = log_loss(IRIS.target[80001:], scikit_predict_proba)
    daal4py_log_loss = log_loss(IRIS.target[80001:], daal4py_predict_proba)
    ratio = daal4py_log_loss / scikit_log_loss
    print("scikit_log_loss :"+str(scikit_log_loss))
    print("daal4py_log_loss :"+str(daal4py_log_loss))
    print("log loss ratio patched:orignal :"+str(ratio))

#    ROC AUC
    scikit_roc_auc = roc_auc_score(IRIS.target[80001:], scikit_predict_proba[:,1], multi_class='ovr')
    daal4py_roc_auc = roc_auc_score(IRIS.target[80001:], daal4py_predict_proba[:,1], multi_class='ovr')
    ratio = daal4py_roc_auc / scikit_roc_auc
    print("scikit_roc_auc :"+str(scikit_roc_auc))
    print("daal4py_roc_auc :"+str(daal4py_roc_auc))
    print("roc_auc ratio patched:orignal :"+str(ratio))
    

def test_warmstart_sklearn_sklearnex_regressor():
    
    scikit_model = ScikitRandomForestRegressor(n_estimators=10,
                                               max_depth=None,
                                               random_state=777,
                                               warm_start=WARM_START_STATUS,)
    daal4py_model = DaalRandomForestRegressor(n_estimators=10,
                                              max_depth=None,
                                              random_state=777,
                                              warm_start=WARM_START_STATUS,)

    curr_time = time.time()
    
    daal4py_model.fit(IRIS.data[:10000], IRIS.target[:10000])
    daal4py_model.n_estimators +=50
    daal4py_model.fit(IRIS.data[10001:20000], IRIS.target[10001:20000])
    daal4py_model.n_estimators +=50
    daal4py_model.fit(IRIS.data[20001:30000], IRIS.target[20001:30000])
    daal4py_model.n_estimators +=50
    daal4py_model.fit(IRIS.data[40001:50000], IRIS.target[40001:50000])
    daal4py_model.n_estimators +=50
    daal4py_model.fit(IRIS.data[50001:60000], IRIS.target[50001:60000])
    daal4py_model.n_estimators +=50
    daal4py_model.fit(IRIS.data[60001:70000], IRIS.target[60001:70000])
    daal4py_model.n_estimators +=50
    daal4py_model.fit(IRIS.data[70001:80000], IRIS.target[70001:80000])
    
    
    
    print(time.time()- curr_time)

    curr_time = time.time()
    
    scikit_model.fit(IRIS.data[:10000], IRIS.target[:10000])
    scikit_model.n_estimators +=50
    scikit_model.fit(IRIS.data[10001:20000], IRIS.target[10001:20000])
    scikit_model.n_estimators +=50
    scikit_model.fit(IRIS.data[20001:30000], IRIS.target[20001:30000])
    scikit_model.n_estimators +=50
    scikit_model.fit(IRIS.data[40001:50000], IRIS.target[40001:50000])
    scikit_model.n_estimators +=50
    scikit_model.fit(IRIS.data[50001:60000], IRIS.target[50001:60000])
    scikit_model.n_estimators +=50
    scikit_model.fit(IRIS.data[60001:70000], IRIS.target[60001:70000])
    scikit_model.n_estimators +=50
    scikit_model.fit(IRIS.data[70001:80000], IRIS.target[70001:80000])
    
    print(time.time()- curr_time)
    scikit_predict = scikit_model.predict(IRIS.data[80001:])
    daal4py_predict = daal4py_model.predict(IRIS.data[80001:])
    scikit_mse = mean_squared_error(scikit_predict, IRIS.target[80001:])
    daal4py_mse = mean_squared_error(daal4py_predict, IRIS.target[80001:])

    ratio = daal4py_mse / scikit_mse
    print("scikit mse: "+str(scikit_mse))
    print("daal4py mse: "+str(daal4py_mse))
    print("mse ratio :"+str(ratio))
    





test_warmstart_sklearn_sklearnex_classifier()
test_warmstart_sklearn_sklearnex_regressor()