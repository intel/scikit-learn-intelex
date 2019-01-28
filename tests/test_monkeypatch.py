
import daal4py.sklearn

def assert_is_patched(mod, cls_name):
    class_module = getattr(mod, cls_name).__module__
    assert class_module.startswith('daal4py')

def test_monkey_patching():
    _tokens = daal4py.sklearn.sklearn_patch_names()
    assert isinstance(_tokens, list) and len(_tokens) > 0
    for t in _tokens:
        daal4py.sklearn.unpatch_sklearn(t)
    for t in _tokens:
        daal4py.sklearn.patch_sklearn(t)

    try:
        import sklearn
        import sklearn.decomposition as skde
        assert_is_patched(skde, 'PCA')
        import sklearn.linear_model as sklm
        assert_is_patched(sklm, 'Ridge')
        assert_is_patched(sklm, 'LinearRegression')
        import sklearn.cluster as skcl
        assert_is_patched(skcl, 'KMeans')
        import sklearn.svm as sksvm
        assert_is_patched(sksvm, 'SVC')
    except ImportError:
        pass
