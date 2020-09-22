"""
The 'daal4py.sklearn.neighbors' module implements the k-nearest neighbors
algorithm.
"""

from .knn import KNeighborsClassifier
from .knn import KNeighborsMixin
from .knn import NearestNeighbors

__all__ = ['KNeighborsClassifier', 'KNeighborsMixin', 'NearestNeighbors']
