from __future__ import print_function

import numpy as np

from scipy import sparse as sp
from sklearn.utils import check_random_state, check_X_y
import warnings

import daal4py
from .daal4py_utils import (make2d, getFPType)


LIBSVM_IMPL = ['c_svc', 'nu_svc', 'one_class', 'epsilon_svr', 'nu_svr']

def _dual_coef_getter(self):
    return self._internal_dual_coef_


def _intercept_getter(self):
    return self._internal_intercept_


def _dual_coef_setter(self, val):
    self._internal_dual_coef_ = val
    if hasattr(self, '_daal_model'):
        del self._daal_model
    if getattr(self, '_daal_fit', False):
        self._daal_fit = False


def _intercept_setter(self, val):
    self._internal_intercept_ = val
    if hasattr(self, '_daal_model'):
        del self._daal_model
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


def _daal4py_fit(self, X, y, kernel):

    if self.C <= 0:
        raise ValueError("C <= 0")

    y = make2d(y)
    num_classes = len(self.classes_)

    if num_classes == 2:
        # Intel(R) DAAL requires binary classes to be 1 and -1. sklearn normalizes
        # the classes to 0 and 1, so we temporarily replace the 0s with -1s.
        y[y == 0] = -1

    X_fptype = getFPType(X)

    kf = _daal4py_kf(kernel, X_fptype, gamma = self._gamma)

    svm_train = daal4py.svm_training(
        fptype=X_fptype,
        C=float(self.C),
        accuracyThreshold=float(self.tol),
        tau=1e-12,
        maxIterations=int(self.max_iter if self.max_iter > 0 else 2**30),
        cacheSize=int(self.cache_size * 1024 * 1024),
        doShrinking=bool(self.shrinking),
        # shrinkingStep=,
        kernel=kf
    )

    if num_classes == 2:
        algo = svm_train
    else:
        algo = daal4py.multi_class_classifier_training(
            nClasses=num_classes,
            fptype=X_fptype,
            accuracyThreshold=float(self.tol),
            method='oneAgainstOne',
            maxIterations=int(self.max_iter if self.max_iter > 0 else 2**30),
            training=svm_train,
            #prediction=svm_predict
        )

    res = algo.compute(X, y)
    model = res.model

    self._daal_model = model

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
        self.n_support_ = np.array([np.sum(indices == -1), np.sum(indices == 1)], dtype=np.int32)
    else:
        self.n_support_ = np.array([np.sum(indices == c) for c in self.classes_], dtype=np.int32)

    self.probA_ = np.empty(0)
    self.probB_ = np.empty(0)

    return


def _daal_std(X):
    """DAAL-based threaded computation of X.std()"""
    fpt = getFPType(X)
    alg = daal4py.low_order_moments(fptype=fpt, method='defaultDense', estimatesToCompute='estimatesMeanVariance')
    ssc = alg.compute(X.reshape(-1,1)).sumSquaresCentered
    return np.sqrt(ssc[0, 0] / X.size)


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
        solver_type = LIBSVM_IMPL.index(self._impl)

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

        if self.gamma in ('scale', 'auto_deprecated'):
            kernel_uses_gamma = (not callable(self.kernel) and self.kernel
                                 not in ('linear', 'precomputed'))
            if kernel_uses_gamma:
                if sparse:
                    # std = sqrt(E[X^2] - E[X]^2)
                    X_std = np.sqrt((X.multiply(X)).mean() - (X.mean())**2)
                else:
                    X_std = _daal_std(X) # X.std()
            else:
                X_std = 1.0 / X.shape[1]
            if self.gamma == 'scale':
                if X_std != 0:
                    self._gamma = 1.0 / (X.shape[1] * X_std)
                else:
                    self._gamma = 1.0
            else:
                if kernel_uses_gamma and not np.isclose(X_std, 1.0):
                    # NOTE: when deprecation ends we need to remove explicitly
                    # setting `gamma` in examples (also in tests). See
                    # https://github.com/scikit-learn/scikit-learn/pull/10331
                    # for the examples/tests that need to be reverted.
                    warnings.warn("The default value of gamma will change "
                                  "from 'auto' to 'scale' in version 0.22 to "
                                  "account better for unscaled features. Set "
                                  "gamma explicitly to 'auto' or 'scale' to "
                                  "avoid this warning.", FutureWarning)
                self._gamma = 1.0 / X.shape[1]
        elif self.gamma == 'auto':
            self._gamma = 1.0 / X.shape[1]
        else:
            self._gamma = self.gamma


        kernel = self.kernel
        if callable(kernel):
            kernel = 'precomputed'

        fit = self._sparse_fit if self._sparse else self._dense_fit
        if self.verbose:  # pragma: no cover
            print('[LibSVM]', end='')

        # see comment on the other call to np.iinfo in this file
        seed = rnd.randint(np.iinfo('i').max)

        if ( not sparse and not self.probability and
             sample_weight.size == 0 and self.class_weight is None and kernel in ['linear', 'rbf']):

            self._daal_fit = True
            _daal4py_fit(self, X, y, kernel)
            self.fit_status_ = 0
        else:
            self._daal_fit = False
            fit(X, y, sample_weight, solver_type, kernel, random_seed=seed)


        self.shape_fit_ = X.shape

        # In binary case, we need to flip the sign of coef, intercept and
        # decision function. Use self._intercept_ and self._dual_coef_ internally.
        if not self._daal_fit:
            self._intercept_ = self.intercept_.copy()
            self._dual_coef_ = self.dual_coef_.copy()
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

    predictionRes = alg.compute(X, self._daal_model)

    res = predictionRes.prediction
    res = res.ravel()

    if num_classes == 2:
        # Convert from Intel(R) DAAL format back to original classes
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
    X = self._validate_for_predict(X)
    if getattr(self, '_daal_fit', False) and hasattr(self, '_daal_model'):
        y = _daal4py_predict(self, X)
    else:
        predict = self._sparse_predict if self._sparse else self._dense_predict
        y = predict(X)

    return self.classes_.take(np.asarray(y, dtype=np.intp))
