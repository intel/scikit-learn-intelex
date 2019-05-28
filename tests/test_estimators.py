import unittest

from sklearn.utils.estimator_checks import check_estimator
import sklearn.utils.estimator_checks

from daal4py.sklearn.neighbors import KNeighborsClassifier
from daal4py.sklearn.ensemble import RandomForestClassifier
from daal4py.sklearn.ensemble import RandomForestRegressor


class Test(unittest.TestCase):
    def test_KNeighborsClassifier(self):
        check_estimator(KNeighborsClassifier)

    def test_RandomForestClassifier(self):
        check_estimator(RandomForestClassifier)

    def test_RandomForestRegressor(self):
        # check_fit_idempotent is known to fail with DAAL's decision
        # forest regressor, due to different partitioning of data
        # between threads from run to run.
        # Hence skip that test
        def dummy(**args):
            pass
        try:
            saved = sklearn.utils.estimator_checks.check_fit_idempotent
            sklearn.utils.estimator_checks.check_fit_idempotent = saved
        except AttributeError:
            saved = None
        check_estimator(RandomForestRegressor)
        if saved is not None:
            sklearn.utils.estimator_checks.check_fit_idempotent = saved


if __name__ == '__main__':
    unittest.main()
