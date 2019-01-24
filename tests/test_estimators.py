import unittest

from sklearn.utils.estimator_checks import check_estimator

from daal4py.sklearn.neighbors import KNeighborsClassifier
from daal4py.sklearn import RandomForestClassifier
from daal4py.sklearn import RandomForestRegressor


class Test(unittest.TestCase):
    def test_KNeighborsClassifier(self):
        check_estimator(KNeighborsClassifier)

    def test_RandomForestClassifier(self):
        check_estimator(RandomForestClassifier)

    def test_RandomForestRegressor(self):
        check_estimator(RandomForestRegressor)


if __name__ == '__main__':
    unittest.main()
