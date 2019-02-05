import unittest
import daal4py.sklearn


class MonkeyPTest(unittest.TestCase):
    def test_monkey_patching(self):
        _tokens = daal4py.sklearn.sklearn_patch_names()
        self.assertTrue(isinstance(_tokens, list) and len(_tokens) > 0)
        for t in _tokens:
            daal4py.sklearn.unpatch_sklearn(t)
        for t in _tokens:
            daal4py.sklearn.patch_sklearn(t)

        import sklearn
        for a in [(sklearn.decomposition, 'PCA'),
                  (sklearn.linear_model, 'Ridge'),
                  (sklearn.linear_model, 'LinearRegression'),
                  (sklearn.cluster, 'KMeans'),
                  (sklearn.svm, 'SVC'),]:
            class_module = getattr(a[0], a[1]).__module__
            self.assertTrue(class_module.startswith('daal4py'))
