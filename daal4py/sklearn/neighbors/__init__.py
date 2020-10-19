"""
The 'daal4py.sklearn.neighbors' module implements the k-nearest neighbors
algorithm.
"""


from ._classification import KNeighborsClassifier
from ._unsupervised import NearestNeighbors
from ._regression import KNeighborsRegressor

__all__ = ['KNeighborsClassifier', 'NearestNeighbors', 'KNeighborsRegressor']
