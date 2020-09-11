"""
The 'daal4py.sklearn.neighbors' module implements the k-nearest neighbors
algorithm.
"""

from .knn import KNeighborsClassifier
from .knn import KNeighborsMixin

__all__ = ['KNeighborsClassifier', 'KNeighborsMixin']
