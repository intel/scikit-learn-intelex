"""
The 'daal4py.sklearn.ensemble' module implements daal4py-based 
RandomForestClassifier and RandomForestRegressor classes.
"""
from daal4py.sklearn._utils import daal_check_version
if daal_check_version((2020, 3), (2021, 9)):
    from .forest import (RandomForestClassifier, RandomForestRegressor)
else:
    from .decision_forest import (RandomForestClassifier, RandomForestRegressor)
from .GBTDAAL import (GBTDAALClassifier, GBTDAALRegressor)
from .AdaBoostClassifier import AdaBoostClassifier

__all__ = ['RandomForestClassifier', 'RandomForestRegressor', 'GBTDAALClassifier', 'GBTDAALRegressor', 'AdaBoostClassifier']

