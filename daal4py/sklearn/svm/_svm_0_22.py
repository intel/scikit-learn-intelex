#===============================================================================
# Copyright 2014-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

from __future__ import print_function

import numpy as np

from scipy import sparse as sp
from sklearn.utils import check_random_state, check_X_y
from sklearn.utils.validation import check_is_fitted, _check_sample_weight

import sklearn.svm._classes as svm_classes
import sklearn.svm._base as svm_base
import warnings

from distutils.version import LooseVersion
from sklearn import __version__ as sklearn_version


import daal4py
from .._utils import (make2d, getFPType, get_patch_message, sklearn_check_version)
import logging

def _get_libsvm_impl():
    return ['c_svc', 'nu_svc', 'one_class', 'epsilon_svr', 'nu_svr']

def _dual_coef_getter(self):
    return self._internal_dual_coef_


def _intercept_getter(self):
    return self._internal_intercept_


def _dual_coef_setter(self, val):
    self._internal_dual_coef_ = val
    if hasattr(self, 'daal_model_'):
        del self.daal_model_
    if getattr(self, '_daal_fit', False):
        self._daal_fit = False


def _intercept_setter(self, val):
    self._internal_intercept_ = val
    if hasattr(self, 'daal_model_'):
        del self.daal_model_
    if getattr(self, '_daal_fit', False):
        self._daal_fit = False


# Methods to extract coefficients
def group_indices_by_class(num_classes, sv_ind_by_clf, labels):
    sv_ind_counters = np.zeros(num_classes, dtype=np.intp)

    num_of_sv_per_class = np.bincount(labels[np.hstack(sv_ind_by_clf)])
    sv_ind_by_class = [np.empty(n, dtype=np.int32) for n in num_of_sv_per_class]

    for indices_per_clf in sv_ind_by_clf:
        for sv_index in indices_per_clf:
            sv_label = labels[sv_index]
            i = sv_ind_counters[sv_label]
            sv_ind_by_class[sv_label][i] = sv_index
            sv_ind_counters[sv_label] += 1

    return sv_ind_by_class


def map_sv_to_columns_in_dual_coef_matrix(sv_ind_by_class):
    from collections import defaultdict
    sv_ind_mapping = defaultdict(lambda: -1)
    p = 0
    for indices_per_class in sv_ind_by_class:
        indices_per_class.sort()
        for sv_index in indices_per_class:
            if sv_ind_mapping[sv_index] == -1:
                sv_ind_mapping[sv_index] = p
                p += 1
    return sv_ind_mapping


def map_to_lexicographic(n):
    """Returns permutation of reverse lexicographics to lexicographics orders for pairs of n consecutive integer indexes"""
    from itertools import (combinations, count)
    two_class_order_gen = ((j, i) for i in range(n) for j in range(i))
    reverse_lookup = { key:val for key,val in zip(two_class_order_gen, count(0))}
    perm_iter = (reverse_lookup[pair] for pair in combinations(range(n), 2))
    return np.fromiter(perm_iter, dtype=np.intp)


def permute_list(li, perm):
    "Rearrange `li` according to `perm`"
    return [ li[i] for i in perm ]


def extract_dual_coef(num_classes, sv_ind_by_clf, sv_coef_by_clf, labels):
    "Construct dual coefficients array in SKLearn peculiar layout, as well corresponding support vector indexes"
    sv_ind_by_class = group_indices_by_class(num_classes, sv_ind_by_clf, labels)
    sv_ind_mapping = map_sv_to_columns_in_dual_coef_matrix(sv_ind_by_class)

    num_unique_sv = len(sv_ind_mapping)
    dc_dt = sv_coef_by_clf[0].dtype

    dual_coef = np.zeros((num_classes - 1, num_unique_sv), dtype=dc_dt)
    support_ = np.empty((num_unique_sv,), dtype=np.int32)

    p = 0
    for i in range(0, num_classes):
        for j in range(i + 1, num_classes):
            sv_ind_i_vs_j = sv_ind_by_clf[p]
            sv_coef_i_vs_j = sv_coef_by_clf[p]
            p += 1

            for k, sv_index in enumerate(sv_ind_i_vs_j):
                label = labels[sv_index]
                col_index = sv_ind_mapping[sv_index]
                if j == label:
                    row_index = i
                else:
                    row_index = j - 1
                dual_coef[row_index, col_index] = sv_coef_i_vs_j[k]
                support_[col_index] = sv_index

    return dual_coef, support_


def _daal4py_kf(kernel, X_fptype, gamma=1.0):
    if kernel == 'rbf':
        sigma_value = np.sqrt(0.5/gamma)
        kf = daal4py.kernel_function_rbf(fptype=X_fptype, sigma=sigma_value)
    elif kernel == 'linear':
        kf = daal4py.kernel_function_linear(fptype=X_fptype)
    else:
        raise ValueError("_daal4py_fit received unexpected kernel specifiction {}.".format(kernel))

    return kf

def _daal4py_check_weight(self, X, y, sample_weight):
    ww = None
    if sample_weight.shape[0] > 0:
        sample_weight = _check_sample_weight(sample_weight, X)
        if np.all(sample_weight <= 0):
            raise ValueError('Invalid input - all samples have zero or negative weights.')
        elif np.any(sample_weight <= 0):
            if len(np.unique(y[sample_weight > 0])) != len(self.classes_):
                raise ValueError('Invalid input - all samples with positive weights have the same label.')
        ww = sample_weight
    elif self.class_weight is not None:
        ww = np.ones(X.shape[0], dtype=np.float64)
    if self.class_weight is not None:
        for i, v in enumerate(self.class_weight_):
            ww[y == i] *= v
    if ww is not None:
        ww = make2d(ww)
    return ww

def _daal4py_svm(fptype, C, accuracyThreshold, tau,
        maxIterations, cacheSize, doShrinking, kernel, nClasses=2):
    svm_train = daal4py.svm_training(
        method='thunder',
        fptype=fptype,
        C=C,
        accuracyThreshold=accuracyThreshold,
        tau=tau,
        maxIterations=maxIterations,
        cacheSize=cacheSize,
        doShrinking=doShrinking,
        kernel=kernel
    )
    if nClasses == 2:
        algo = svm_train
    else:
        algo = daal4py.multi_class_classifier_training(
            nClasses=nClasses,
            fptype=fptype,
            method='oneAgainstOne',
            training=svm_train,
        )

    return algo


def _daal4py_fit(self, X, y_inp, sample_weight, kernel):

    if self.C <= 0:
        raise ValueError("C <= 0")
    num_classes = len(self.classes_)

    ww = _daal4py_check_weight(self, X, y_inp, sample_weight)

    y = make2d(y_inp)
    if num_classes == 2:
        # Intel(R) oneAPI Data Analytics Library requires binary classes to
        # be 1 and -1. sklearn normalizes the classes to 0 and 1, so we
        # temporarily replace the 0s with -1s.
        y = y.copy()
        y[y == 0] = -1

    X_fptype = getFPType(X)

    kf = _daal4py_kf(kernel, X_fptype, gamma = self._gamma)
    algo = _daal4py_svm(fptype=X_fptype,
        C=float(self.C),
        accuracyThreshold=float(self.tol),
        tau=1e-12,
        maxIterations=int(self.max_iter if self.max_iter > 0 else 2**30),
        cacheSize=int(self.cache_size * 1024 * 1024),
        doShrinking=bool(self.shrinking),
        kernel=kf,
        nClasses=num_classes)

    res = algo.compute(data=X, labels=y, weights=ww)
    model = res.model
    self.daal_model_ = model

    if num_classes == 2:
        # binary
        two_class_sv_ind_ = model.SupportIndices
        two_class_sv_ind_ = two_class_sv_ind_.ravel()

        # support indexes need permutation to arrange them into the same layout as that of Scikit-Learn
        tmp = np.empty(two_class_sv_ind_.shape, dtype=np.dtype([('label', y.dtype), ('ind', two_class_sv_ind_.dtype)]))
        tmp['label'][:] = y[two_class_sv_ind_].ravel()
        tmp['ind'][:] = two_class_sv_ind_
        perm = np.argsort(tmp, order=['label', 'ind'])
        del tmp

        self.support_ = two_class_sv_ind_[perm]
        self.support_vectors_ = X[self.support_]

        self.dual_coef_ = model.ClassificationCoefficients.T
        self.dual_coef_ = self.dual_coef_[:, perm]
        self.intercept_ = np.array([model.Bias])

    else:
        # multi-class
        intercepts = []
        coefs = []
        num_models = model.NumberOfTwoClassClassifierModels
        sv_ind_by_clf = []
        label_indexes = []

        model_id = 0
        for i1 in range(num_classes):
            label_indexes.append(np.where( y == i1 )[0])
            for i2 in range(i1):
                svm_model = model.TwoClassClassifierModel(model_id)

                # Indices correspond to input features with label i1 followed by input features with label i2
                two_class_sv_ind_ = svm_model.SupportIndices
                # Map these indexes to indexes of the training data
                sv_ind = np.take(np.hstack((label_indexes[i1], label_indexes[i2])), two_class_sv_ind_.ravel())
                sv_ind_by_clf.append(sv_ind)

                # svs_ = getArrayFromNumericTable(svm_model.getSupportVectors())
                # assert np.array_equal(svs_, X[sv_ind])

                intercepts.append(-svm_model.Bias)
                coefs.append(-svm_model.ClassificationCoefficients)
                model_id += 1

        # permute solutions to lexicographic ordering
        to_lex_perm = map_to_lexicographic(num_classes)
        sv_ind_by_clf = permute_list(sv_ind_by_clf, to_lex_perm)
        sv_coef_by_clf = permute_list(coefs, to_lex_perm)
        intercepts = permute_list(intercepts, to_lex_perm)

        self.dual_coef_, self.support_ = extract_dual_coef(
            num_classes,    # number of classes
            sv_ind_by_clf,  # support vector indexes by two-class classifiers
            sv_coef_by_clf, # classification coefficients by two-class classifiers
            y.squeeze().astype(np.intp, copy=False)   # integer labels
        )
        self.support_vectors_ = X[self.support_]
        self.intercept_ = np.array(intercepts)

    indices = y.take(self.support_, axis=0)
    if num_classes == 2:
        self._n_support = np.array([np.sum(indices == -1), np.sum(indices == 1)], dtype=np.int32)
    else:
        self._n_support = np.array([np.sum(indices == i) for i, c in enumerate(self.classes_)], dtype=np.int32)

    try:
        self.probA_ = np.empty(0)
        self.probB_ = np.empty(0)
    except AttributeError:
        # in master probA_ and probB_ are deprecated read-only attributes
        self._probA = np.empty(0)
        self._probB = np.empty(0)


def __compute_gamma__(gamma, kernel, X, sparse, use_var=True, deprecation=True):
    """
    Computes actual value of 'gamma' parameter of RBF kernel
    corresponding to SVC keyword values `gamma` and `kernel`, and feature
    matrix X, with sparsity `sparse`.

    In 0.20 gamma='scale' used to mean compute 'gamma' based on
    column-wise standard deviation, but in 0.20.3 it was changed
    to use column-wise variance.

    See: https://github.com/scikit-learn/scikit-learn/pull/13221
    """
    if deprecation:
        _gamma_is_scale = gamma in ('scale', 'auto_deprecated')
    else:
        _gamma_is_scale = (gamma == 'scale')
    if _gamma_is_scale:
        kernel_uses_gamma = (not callable(kernel) and kernel
                             not in ('linear', 'precomputed'))
        if kernel_uses_gamma:
            if sparse:
                # var = E[X^2] - E[X]^2
                X_sc = (X.multiply(X)).mean() - (X.mean())**2
            else:
                X_sc = X.var()
            if not use_var:
                X_sc = np.sqrt(X_sc)
        else:
            X_sc = 1.0 / X.shape[1]
        if gamma == 'scale':
            if X_sc != 0:
                _gamma = 1.0 / (X.shape[1] * X_sc)
            else:
                _gamma = 1.0
        else:
            if kernel_uses_gamma and deprecation and not np.isclose(X_sc, 1.0):
                # NOTE: when deprecation ends we need to remove explicitly
                # setting `gamma` in examples (also in tests). See
                # https://github.com/scikit-learn/scikit-learn/pull/10331
                # for the examples/tests that need to be reverted.
                warnings.warn("The default value of gamma will change "
                              "from 'auto' to 'scale' in version 0.22 to "
                              "account better for unscaled features. Set "
                              "gamma explicitly to 'auto' or 'scale' to "
                              "avoid this warning.", FutureWarning)
            _gamma = 1.0 / X.shape[1]
    elif gamma == 'auto':
        _gamma = 1.0 / X.shape[1]
    elif isinstance(gamma, str) and not deprecation:
        raise ValueError(
            "When 'gamma' is a string, it should be either 'scale' or "
            "'auto'. Got '{}' instead.".format(gamma)
        )
    else:
        _gamma = gamma

    return _gamma

def _compute_gamma(*args):
    no_older_than_0_20_3 = sklearn_check_version("0.20.3")
    no_older_than_0_22 = not sklearn_check_version("0.22")
    return __compute_gamma__(*args, use_var=no_older_than_0_20_3, deprecation=no_older_than_0_22)


def fit(self, X, y, sample_weight=None):
        """Fit the SVM model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
            For kernel="precomputed", the expected shape of X is
            (n_samples, n_samples).

        y : array-like, shape (n_samples,)
            Target values (class labels in classification, real numbers in
            regression)

        sample_weight : array-like, shape (n_samples,)
            Per-sample weights. Rescale C per sample. Higher weights
            force the classifier to put more emphasis on these points.

        Returns
        -------
        self : object

        Notes
        ------
        If X and y are not C-ordered and contiguous arrays of np.float64 and
        X is not a scipy.sparse.csr_matrix, X and/or y may be copied.

        If X is a dense array, then the other methods will not support sparse
        matrices as input.
        """
        rnd = check_random_state(self.random_state)

        sparse = sp.isspmatrix(X)
        if sparse and self.kernel == "precomputed":
            raise TypeError("Sparse precomputed kernels are not supported.")
        self._sparse = sparse and not callable(self.kernel)

        X, y = check_X_y(X, y, dtype=np.float64, order='C', accept_sparse='csr',
                         accept_large_sparse=False)
        y = self._validate_targets(y)

        sample_weight = np.asarray([]
                                   if sample_weight is None
                                   else sample_weight, dtype=np.float64)
        solver_type = _get_libsvm_impl().index(self._impl)

        # input validation
        if solver_type != 2 and X.shape[0] != y.shape[0]:
            raise ValueError("X and y have incompatible shapes.\n" +
                             "X has %s samples, but y has %s." %
                             (X.shape[0], y.shape[0]))

        if self.kernel == "precomputed" and X.shape[0] != X.shape[1]:
            raise ValueError("X.shape[0] should be equal to X.shape[1]")

        if sample_weight.shape[0] > 0 and sample_weight.shape[0] != X.shape[0]:
            raise ValueError("sample_weight and X have incompatible shapes: "
                             "%r vs %r\n"
                             "Note: Sparse matrices cannot be indexed w/"
                             "boolean masks (use `indices=True` in CV)."
                             % (sample_weight.shape, X.shape))

        self._gamma = _compute_gamma(self.gamma, self.kernel, X, sparse)

        kernel = self.kernel
        if callable(kernel):
            kernel = 'precomputed'

        fit = self._sparse_fit if self._sparse else self._dense_fit
        if self.verbose:  # pragma: no cover
            print('[LibSVM]', end='')

        # see comment on the other call to np.iinfo in this file
        seed = rnd.randint(np.iinfo('i').max)
        if ( not sparse and not self.probability and not getattr(self, 'break_ties', False) and \
             kernel in ['linear', 'rbf']):

            logging.info("sklearn.svm.SVC.fit: " + get_patch_message("daal"))
            self._daal_fit = True
            _daal4py_fit(self, X, y, sample_weight, kernel)
            self.fit_status_ = 0
        else:
            logging.info("sklearn.svm.SVC.fit: " + get_patch_message("sklearn"))
            self._daal_fit = False
            fit(X, y, sample_weight, solver_type, kernel, random_seed=seed)


        self.shape_fit_ = X.shape

        # In binary case, we need to flip the sign of coef, intercept and
        # decision function. Use self._intercept_ and self._dual_coef_ internally.
        if not self._daal_fit:
            self._internal_intercept_ = self.intercept_.copy()
            self._internal_dual_coef_ = self.dual_coef_.copy()
        else:
            self._internal_intercept_ = self.intercept_.copy()
            self._internal_dual_coef_ = self.dual_coef_.copy()
            if len(self.classes_) == 2:
                self._internal_dual_coef_ *= -1
                self._internal_intercept_ *= -1


        if not self._daal_fit and len(self.classes_) == 2 and self._impl in ['c_svc', 'nu_svc']:
            self.intercept_ *= -1
            self.dual_coef_ *= -1

        return self


def _daal4py_predict(self, X):
    X_fptype = getFPType(X)
    num_classes = len(self.classes_)

    kf = _daal4py_kf(self.kernel, X_fptype, gamma = self._gamma)

    svm_predict = daal4py.svm_prediction(
        fptype=X_fptype,
        method='defaultDense',
        kernel=kf
    )
    if num_classes == 2:
        alg = svm_predict
    else:
        alg = daal4py.multi_class_classifier_prediction(
            nClasses=num_classes,
            fptype=X_fptype,
            maxIterations=int(self.max_iter if self.max_iter > 0 else 2**30),
            accuracyThreshold=float(self.tol),
            pmethod="voteBased",
            tmethod='oneAgainstOne',
            prediction=svm_predict
        )

    predictionRes = alg.compute(X, self.daal_model_)

    res = predictionRes.prediction
    res = res.ravel()

    if num_classes == 2:
        # Convert from Intel(R) oneAPI Data Analytics Library format back to original classes
        np.greater(res, 0, out=res)

    return res


def predict(self, X):
    """Perform regression on samples in X.

    For an one-class model, +1 (inlier) or -1 (outlier) is returned.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        For kernel="precomputed", the expected shape of X is
        (n_samples_test, n_samples_train).

    Returns
    -------
    y_pred : array, shape (n_samples,)
    """
    check_is_fitted(self)
    _break_ties = getattr(self, 'break_ties', False)
    if _break_ties and self.decision_function_shape == 'ovo':
        raise ValueError("break_ties must be False when "
                         "decision_function_shape is 'ovo'")

    if (_break_ties
        and self.decision_function_shape == 'ovr'
        and len(self.classes_) > 2):
        logging.info("sklearn.svm.SVC.predict: " + get_patch_message("sklearn"))
        y = np.argmax(self.decision_function(X), axis=1)
    else:
        X = self._validate_for_predict(X)
        if getattr(self, '_daal_fit', False) and hasattr(self, 'daal_model_'):
            logging.info("sklearn.svm.SVC.predict: " + get_patch_message("daal"))
            y = _daal4py_predict(self, X)
        else:
            logging.info("sklearn.svm.SVC.predict: " + get_patch_message("sklearn"))
            predict_func = self._sparse_predict if self._sparse else self._dense_predict
            y = predict_func(X)

    return self.classes_.take(np.asarray(y, dtype=np.intp))


__base_svc_init_arg_names__ = []

__base_svc_init_function__ = svm_base.BaseSVC.__init__
__base_svc_init_function_code__ = __base_svc_init_function__.__code__

try:
    # retrieve tuple of code argument names to check whether
    # new in 0.22 keyword 'break_ties' is in it
    __base_svc_init_arg_names__ = __base_svc_init_function_code__.co_varnames
except AttributeError:
    pass

del __base_svc_init_function__
del __base_svc_init_function_code__


if 'break_ties' in  __base_svc_init_arg_names__:
    class SVC(svm_base.BaseSVC):
        _impl = 'c_svc'

        def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale',
                     coef0=0.0, shrinking=True, probability=False,
                     tol=1e-3, cache_size=200, class_weight=None,
                     verbose=False, max_iter=-1, decision_function_shape='ovr',
                     break_ties=False, random_state=None):

            super(SVC, self).__init__(
                kernel=kernel, degree=degree, gamma=gamma,
                coef0=coef0, tol=tol, C=C, nu=0., shrinking=shrinking,
                probability=probability, cache_size=cache_size,
                class_weight=class_weight, verbose=verbose, max_iter=max_iter,
                decision_function_shape=decision_function_shape, break_ties=break_ties,
                random_state=random_state)
else:
    class SVC(svm_base.BaseSVC):
        _impl = 'c_svc'

        def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='auto_deprecated',
                     coef0=0.0, shrinking=True, probability=False,
                     tol=1e-3, cache_size=200, class_weight=None,
                     verbose=False, max_iter=-1, decision_function_shape='ovr',
                     random_state=None):

            super(SVC, self).__init__(
                kernel=kernel, degree=degree, gamma=gamma,
                coef0=coef0, tol=tol, C=C, nu=0., shrinking=shrinking,
                probability=probability, cache_size=cache_size,
                class_weight=class_weight, verbose=verbose, max_iter=max_iter,
                decision_function_shape=decision_function_shape,
                random_state=random_state)

SVC.fit = fit
SVC.predict = predict
SVC._dual_coef_ = property(_dual_coef_getter, _dual_coef_setter)
SVC._intercept_ = property(_intercept_getter, _intercept_setter)
SVC.__doc__ = svm_classes.SVC.__doc__
