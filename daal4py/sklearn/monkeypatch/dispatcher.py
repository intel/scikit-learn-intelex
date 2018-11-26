from sklearn import __version__ as sklearn_version
from distutils.version import LooseVersion

from sklearn.decomposition.pca import PCA
from sklearn.cluster.k_means_ import KMeans
from sklearn.metrics import pairwise
from sklearn.linear_model.base import LinearRegression
from sklearn.linear_model.ridge import Ridge
from sklearn.svm.classes import SVC
import sklearn.linear_model.logistic as logistic_module

from .k_means import fit as kmeans_fit
from .k_means import predict as kmeans_predict
from .pca import _fit_full as pca_fit_full
from .pairwise import daal_pairwise_distances
from .linear import fit as linear_fit
from .linear import predict as linear_predict
from .ridge import fit as ridge_fit
from .ridge import predict as ridge_predict
from .svm import fit as svm_fit
from .svm import predict as svm_predict
from .svm import _dual_coef_setter as _internal_dual_coef_setter
from .svm import _dual_coef_getter as _internal_dual_coef_getter
from .svm import _intercept_setter as _internal_intercept_setter
from .svm import _intercept_getter as _internal_intercept_getter
from .logistic_path import logistic_regression_path as daal_optimized_logistic_path

_mapping = {
    'pca_full':  [[(PCA, '_fit_full', pca_fit_full), None]],
    'kmeans':    [[(KMeans, 'fit', kmeans_fit), None], [(KMeans, 'predict', kmeans_predict), None]],
    'distances': [[(pairwise, 'pairwise_distances', daal_pairwise_distances), None]],
    'linear':    [[(LinearRegression, 'fit', linear_fit), None],
                  [(LinearRegression, 'predict', linear_predict), None]],
    'ridge':     [[(Ridge, 'fit', ridge_fit), None],
                  [(Ridge, 'predict', ridge_predict), None]],
    'svm':       [[(SVC, 'fit', svm_fit), None],
                  [(SVC, 'predict', svm_predict), None],
                  [(SVC, '_dual_coef_', property(_internal_dual_coef_getter, _internal_dual_coef_setter)), None],
                  [(SVC, '_intercept_', property(_internal_intercept_getter, _internal_intercept_setter)), None]],
    'logistic':  [[(logistic_module, 'logistic_regression_path', daal_optimized_logistic_path), None]],
}


def do_patch(name):
    lname = name.lower()
    if lname in _mapping:
        for descriptor in _mapping[lname]:
            which, what, replacer = descriptor[0]
            if descriptor[1] is None:
                descriptor[1] = getattr(which, what, None)
            setattr(which, what, replacer)
    else:
        raise ValueError("Has no patch for: " + name)


def do_unpatch(name):
    lname = name.lower()
    if lname in _mapping:
        for descriptor in _mapping[lname]:
            which, what, replacer = descriptor[0]
            setattr(which, what, descriptor[1])
    else:
        raise ValueError("Has no patch for: " + name)


def enable(name=None):
    if LooseVersion(sklearn_version) < LooseVersion("0.20.0"):
        raise NotImplementedError("daal4sklearn is for scikit-learn 0.20.0 only ...")
    elif LooseVersion(sklearn_version) > LooseVersion("0.20.1"):
        warnings.warn("daal4sklearn {daal4py_version} has only been tested with scikit-learn 0.20.0, found version...")

    if name is not None:
        do_patch(name)
    else:
        for key in _mapping:
            do_patch(key)


def disable(name=None):
    if name is not None:
        do_unpatch(name)
    else:
        for key in _mapping:
            do_unpatch(key)


def _patch_names():
    return list(_mapping.keys())
