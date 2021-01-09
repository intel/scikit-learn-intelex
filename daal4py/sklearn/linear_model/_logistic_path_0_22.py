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

import daal4py as d4p
import numpy as np
import scipy.sparse as sparse
import scipy.optimize as optimize
import numbers
import warnings

from .logistic_loss import (_daal4py_loss_and_grad,
                            _daal4py_logistic_loss_extra_args,
                            _daal4py_cross_entropy_loss_extra_args,
                            _daal4py_loss_, _daal4py_grad_,
                            _daal4py_grad_hess_)

from sklearn import __version__ as sklearn_version
from distutils.version import LooseVersion
from sklearn.utils import (check_array,
                           check_consistent_length,
                           compute_class_weight,
                           check_random_state)
from sklearn.utils.validation import _check_sample_weight, check_is_fitted
from sklearn.linear_model._sag import sag_solver
from sklearn.utils.optimize import _newton_cg, _check_optimize_result
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model._logistic import (
    _check_solver,
    _check_multi_class,
    _fit_liblinear,
    _logistic_loss_and_grad,
    _logistic_loss,
    _logistic_grad_hess,
    _multinomial_loss,
    _multinomial_loss_grad,
    _multinomial_grad_hess,
    _LOGISTIC_SOLVER_CONVERGENCE_MSG,
    LogisticRegression as LogisticRegression_original)
from sklearn.preprocessing import (LabelEncoder, LabelBinarizer)
from sklearn.linear_model._base import (LinearClassifierMixin, SparseCoefMixin, BaseEstimator)
from .._utils import (daal_check_version, getFPType,
                      get_patch_message)
import logging

# Code adapted from sklearn.linear_model.logistic prior to 0.21
def logistic_regression_path(X, y, pos_class=None, Cs=10, fit_intercept=True,
                             max_iter=100, tol=1e-4, verbose=0,
                             solver='lbfgs', coef=None,
                             class_weight=None, dual=False, penalty='l2',
                             intercept_scaling=1., multi_class='warn',
                             random_state=None, check_input=True,
                             max_squared_sum=None, sample_weight=None):
    """Compute a Logistic Regression model for a list of regularization
    parameters.

    This is an implementation that uses the result of the previous model
    to speed up computations along the set of solutions, making it faster
    than sequentially calling LogisticRegression for the different parameters.
    Note that there will be no speedup with liblinear solver, since it does
    not handle warm-starting.

    Read more in the :ref:`User Guide <logistic_regression>`.

    Parameters
    ----------
    X : array-like or sparse matrix, shape (n_samples, n_features)
        Input data.

    y : array-like, shape (n_samples,) or (n_samples, n_targets)
        Input data, target values.

    pos_class : int, None
        The class with respect to which we perform a one-vs-all fit.
        If None, then it is assumed that the given problem is binary.

    Cs : int | array-like, shape (n_cs,)
        List of values for the regularization parameter or integer specifying
        the number of regularization parameters that should be used. In this
        case, the parameters will be chosen in a logarithmic scale between
        1e-4 and 1e4.

    fit_intercept : bool
        Whether to fit an intercept for the model. In this case the shape of
        the returned array is (n_cs, n_features + 1).

    max_iter : int
        Maximum number of iterations for the solver.

    tol : float
        Stopping criterion. For the newton-cg and lbfgs solvers, the iteration
        will stop when ``max{|g_i | i = 1, ..., n} <= tol``
        where ``g_i`` is the i-th component of the gradient.

    verbose : int
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.

    solver : {'lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'}
        Numerical solver to use.

    coef : array-like, shape (n_features,), default None
        Initialization value for coefficients of logistic regression.
        Useless for liblinear solver.

    class_weight : dict or 'balanced', optional
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    dual : bool
        Dual or primal formulation. Dual formulation is only implemented for
        l2 penalty with liblinear solver. Prefer dual=False when
        n_samples > n_features.

    penalty : str, 'l1' or 'l2'
        Used to specify the norm used in the penalization. The 'newton-cg',
        'sag' and 'lbfgs' solvers support only l2 penalties.

    intercept_scaling : float, default 1.
        Useful only when the solver 'liblinear' is used
        and self.fit_intercept is set to True. In this case, x becomes
        [x, self.intercept_scaling],
        i.e. a "synthetic" feature with constant value equal to
        intercept_scaling is appended to the instance vector.
        The intercept becomes ``intercept_scaling * synthetic_feature_weight``.

        Note! the synthetic feature weight is subject to l1/l2 regularization
        as all other features.
        To lessen the effect of regularization on synthetic feature weight
        (and therefore on the intercept) intercept_scaling has to be increased.

    multi_class : str, {'ovr', 'multinomial', 'auto'}, default: 'ovr'
        If the option chosen is 'ovr', then a binary problem is fit for each
        label. For 'multinomial' the loss minimised is the multinomial loss fit
        across the entire probability distribution, *even when the data is
        binary*. 'multinomial' is unavailable when solver='liblinear'.
        'auto' selects 'ovr' if the data is binary, or if solver='liblinear',
        and otherwise selects 'multinomial'.

        .. versionadded:: 0.18
           Stochastic Average Gradient descent solver for 'multinomial' case.
        .. versionchanged:: 0.20
            Default will change from 'ovr' to 'auto' in 0.22.

    random_state : int, RandomState instance or None, optional, default None
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`. Used when ``solver`` == 'sag' or
        'liblinear'.

    check_input : bool, default True
        If False, the input arrays X and y will not be checked.

    max_squared_sum : float, default None
        Maximum squared sum of X over samples. Used only in SAG solver.
        If None, it will be computed, going through all the samples.
        The value should be precomputed to speed up cross validation.

    sample_weight : array-like, shape(n_samples,) optional
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight.

    Returns
    -------
    coefs : ndarray, shape (n_cs, n_features) or (n_cs, n_features + 1)
        List of coefficients for the Logistic Regression model. If
        fit_intercept is set to True then the second dimension will be
        n_features + 1, where the last item represents the intercept. For
        ``multiclass='multinomial'``, the shape is (n_classes, n_cs,
        n_features) or (n_classes, n_cs, n_features + 1).

    Cs : ndarray
        Grid of Cs used for cross-validation.

    n_iter : array, shape (n_cs,)
        Actual number of iteration for each Cs.

    Notes
    -----
    You might get slightly different results with the solver liblinear than
    with the others since this uses LIBLINEAR which penalizes the intercept.

    .. versionchanged:: 0.19
        The "copy" parameter was removed.
    """

    if isinstance(Cs, numbers.Integral):
        Cs = np.logspace(-4, 4, Cs)

    solver = _check_solver(solver, penalty, dual)

    # Preprocessing.
    if check_input:
        X = check_array(X, accept_sparse='csr', dtype=np.float64,
                        accept_large_sparse=solver != 'liblinear')
        y = check_array(y, ensure_2d=False, dtype=None)
        check_consistent_length(X, y)
    _, n_features = X.shape

    classes = np.unique(y)
    random_state = check_random_state(random_state)

    multi_class = _check_multi_class(multi_class, solver, len(classes))
    if pos_class is None and multi_class != 'multinomial':
        if (classes.size > 2):
            raise ValueError('To fit OvR, use the pos_class argument')
        # np.unique(y) gives labels in sorted order.
        pos_class = classes[1]

    daal_ready = solver in ['lbfgs', 'newton-cg'] and not sparse.issparse(X)
    daal_ready = daal_ready and sample_weight is None and class_weight is None

    # If sample weights exist, convert them to array (support for lists)
    # and check length
    # Otherwise set them to 1 for all examples
    if not daal_ready:
        if sample_weight is not None:
            sample_weight = np.array(sample_weight, dtype=X.dtype, order='C')
            check_consistent_length(y, sample_weight)
            default_weights = False
        else:
            default_weights = (class_weight is None)
            sample_weight = np.ones(X.shape[0], dtype=X.dtype)

    # If class_weights is a dict (provided by the user), the weights
    # are assigned to the original labels. If it is "balanced", then
    # the class_weights are assigned after masking the labels with a OvR.
    le = LabelEncoder()
    if (isinstance(class_weight, dict) or multi_class == 'multinomial') and not daal_ready:
        class_weight_ = compute_class_weight(class_weight, classes, y)
        if not np.allclose(class_weight_, np.ones_like(class_weight_)):
            sample_weight *= class_weight_[le.fit_transform(y)]

    # For doing a ovr, we need to mask the labels first. for the
    # multinomial case this is not necessary.
    if multi_class == 'ovr':
        mask_classes = np.array([-1, 1])
        mask = (y == pos_class)
        y_bin = np.ones(y.shape, dtype=X.dtype)
        y_bin[~mask] = -1.
        # for compute_class_weight

        if class_weight == "balanced" and not daal_ready:
            class_weight_ = compute_class_weight(class_weight, mask_classes,
                                                 y_bin)
            if not np.allclose(class_weight_, np.ones_like(class_weight_)):
                sample_weight *= class_weight_[le.fit_transform(y_bin)]

        if daal_ready:
            w0 = np.zeros(n_features + 1, dtype=X.dtype)
            y_bin[~mask] = 0.
        else:
            w0 = np.zeros(n_features + int(fit_intercept), dtype=X.dtype)

    else:
        if solver not in ['sag', 'saga']:
            if daal_ready:
                Y_multi = le.fit_transform(y).astype(X.dtype, copy=False)
            else:
                lbin = LabelBinarizer()
                Y_multi = lbin.fit_transform(y)
                if Y_multi.shape[1] == 1:
                    Y_multi = np.hstack([1 - Y_multi, Y_multi])
        else:
            # SAG multinomial solver needs LabelEncoder, not LabelBinarizer
            Y_multi = le.fit_transform(y).astype(X.dtype, copy=False)

        if daal_ready:
            w0 = np.zeros((classes.size, n_features + 1),
                      order='C', dtype=X.dtype)
        else:
            w0 = np.zeros((classes.size, n_features + int(fit_intercept)),
                      order='F', dtype=X.dtype)

    if coef is not None:
        # it must work both giving the bias term and not
        if multi_class == 'ovr':
            if coef.size not in (n_features, w0.size):
                raise ValueError(
                    'Initialization coef is of shape %d, expected shape '
                    '%d or %d' % (coef.size, n_features, w0.size))
            if daal_ready:
                w0[-coef.size:] = np.roll(coef, 1, -1) if coef.size != n_features else coef
            else:
                w0[:coef.size] = coef
        else:
            # For binary problems coef.shape[0] should be 1, otherwise it
            # should be classes.size.
            n_classes = classes.size
            if n_classes == 2:
                n_classes = 1

            if (coef.shape[0] != n_classes or
                    coef.shape[1] not in (n_features, n_features + 1)):
                raise ValueError(
                    'Initialization coef is of shape (%d, %d), expected '
                    'shape (%d, %d) or (%d, %d)' % (
                        coef.shape[0], coef.shape[1], classes.size,
                        n_features, classes.size, n_features + 1))

            if daal_ready:
                w0[:, -coef.shape[1]:] = np.roll(coef, 1, -1) if coef.shape[1] != n_features else coef
            else:
                if n_classes == 1:
                    w0[0, :coef.shape[1]] = -coef
                    w0[1, :coef.shape[1]] = coef
                else:
                    w0[:, :coef.shape[1]] = coef

    C_daal_multiplier = 1
    # commented out because this is Py3 feature
    #def _map_to_binary_logistic_regression():
    #    nonlocal C_daal_multiplier
    #    nonlocal w0
    #    C_daal_multiplier = 2
    #    w0 *= 2

    if multi_class == 'multinomial':
        # fmin_l_bfgs_b and newton-cg accepts only ravelled parameters.
        if solver in ['lbfgs', 'newton-cg']:
            if daal_ready and classes.size == 2:
                w0_saved = w0
                w0 = w0[-1:, :]
            w0 = w0.ravel()
        target = Y_multi
        if solver == 'lbfgs':
            if daal_ready:
                if classes.size == 2:
                    # _map_to_binary_logistic_regression()
                    C_daal_multiplier = 2
                    w0 *= 2
                    daal_extra_args_func = _daal4py_logistic_loss_extra_args
                else:
                    daal_extra_args_func = _daal4py_cross_entropy_loss_extra_args
                func = _daal4py_loss_and_grad
            else:
                func = lambda x, *args: _multinomial_loss_grad(x, *args)[0:2]
        elif solver == 'newton-cg':
            if daal_ready:
                if classes.size == 2:
                    # _map_to_binary_logistic_regression()
                    C_daal_multiplier = 2
                    w0 *= 2
                    daal_extra_args_func = _daal4py_logistic_loss_extra_args
                else:
                    daal_extra_args_func = _daal4py_cross_entropy_loss_extra_args
                func = _daal4py_loss_
                grad = _daal4py_grad_
                hess = _daal4py_grad_hess_
            else:
                func = lambda x, *args: _multinomial_loss(x, *args)[0]
                grad = lambda x, *args: _multinomial_loss_grad(x, *args)[1]
                hess = _multinomial_grad_hess
        warm_start_sag = {'coef': w0.T}
    else:
        target = y_bin
        if solver == 'lbfgs':
            if daal_ready:
                func = _daal4py_loss_and_grad
                daal_extra_args_func = _daal4py_logistic_loss_extra_args
            else:
                func = _logistic_loss_and_grad
        elif solver == 'newton-cg':
            if daal_ready:
                daal_extra_args_func = _daal4py_logistic_loss_extra_args
                func = _daal4py_loss_
                grad = _daal4py_grad_
                hess = _daal4py_grad_hess_
            else:
                func = _logistic_loss
                grad = lambda x, *args: _logistic_loss_and_grad(x, *args)[1]
                hess = _logistic_grad_hess
        warm_start_sag = {'coef': np.expand_dims(w0, axis=1)}

    coefs = list()
    n_iter = np.zeros(len(Cs), dtype=np.int32)
    for i, C in enumerate(Cs):
        if solver == 'lbfgs':
            if daal_ready:
                extra_args = daal_extra_args_func(classes.size, w0, X, target, 0., 0.5 / C / C_daal_multiplier,
                                                  fit_intercept, value=True, gradient=True, hessian=False)
            else:
                extra_args = (X, target, 1. / C, sample_weight)

            iprint = [-1, 50, 1, 100, 101][
                np.searchsorted(np.array([0, 1, 2, 3]), verbose)]
            w0, loss, info = optimize.fmin_l_bfgs_b(
                func, w0, fprime=None,
                args=extra_args,
                iprint=iprint, pgtol=tol, maxiter=max_iter)
            if daal_ready and C_daal_multiplier == 2:
                w0 *= 0.5
            if info["warnflag"] == 1:
                warnings.warn("lbfgs failed to converge. Increase the number "
                              "of iterations.", ConvergenceWarning)
            # In scipy <= 1.0.0, nit may exceed maxiter.
            # See https://github.com/scipy/scipy/issues/7854.
            n_iter_i = min(info['nit'], max_iter)
        elif solver == 'newton-cg':
            if daal_ready:
                def make_ncg_funcs(f, value=False, gradient=False, hessian=False):
                    daal_penaltyL2 = 0.5 / C / C_daal_multiplier
                    _obj_, X_, y_, n_samples = daal_extra_args_func(
                        classes.size, w0, X, target, 0., daal_penaltyL2, fit_intercept,
                        value=value, gradient=gradient, hessian=hessian)
                    _func_ = lambda x, *args: f(x, _obj_, *args)
                    return _func_, (X_, y_, n_samples, daal_penaltyL2)

                loss_func, extra_args  = make_ncg_funcs(func, value=True)
                grad_func, _  = make_ncg_funcs(grad, gradient=True)
                grad_hess_func, _ = make_ncg_funcs(hess, gradient=True)
                w0, n_iter_i = _newton_cg(grad_hess_func, loss_func, grad_func, w0, args=extra_args,
                                         maxiter=max_iter, tol=tol)
            else:
                args = (X, target, 1. / C, sample_weight)
                w0, n_iter_i = _newton_cg(hess, func, grad, w0, args=args,
                                         maxiter=max_iter, tol=tol)
        elif solver == 'liblinear':
            coef_, intercept_, n_iter_i, = _fit_liblinear(
                X, target, C, fit_intercept, intercept_scaling, None,
                penalty, dual, verbose, max_iter, tol, random_state,
                sample_weight=sample_weight)
            if fit_intercept:
                w0 = np.concatenate([coef_.ravel(), intercept_])
            else:
                w0 = coef_.ravel()

        elif solver in ['sag', 'saga']:
            if multi_class == 'multinomial':
                target = target.astype(np.float64)
                loss = 'multinomial'
            else:
                loss = 'log'
            if penalty == 'l1':
                alpha = 0.
                beta = 1. / C
            else:
                alpha = 1. / C
                beta = 0.
            w0, n_iter_i, warm_start_sag = sag_solver(
                X, target, sample_weight, loss, alpha,
                beta, max_iter, tol,
                verbose, random_state, False, max_squared_sum, warm_start_sag,
                is_saga=(solver == 'saga'))

        else:
            raise ValueError("solver must be one of {'liblinear', 'lbfgs', "
                             "'newton-cg', 'sag'}, got '%s' instead" % solver)

        if multi_class == 'multinomial':
            if daal_ready:
                if classes.size == 2:
                   multi_w0 = w0[np.newaxis, :]
                else:
                   multi_w0 = np.reshape(w0, (classes.size, -1))
            else:
                n_classes = max(2, classes.size)
                multi_w0 = np.reshape(w0, (n_classes, -1))
                if n_classes == 2:
                    multi_w0 = multi_w0[1][np.newaxis, :]
            coefs.append(np.require(multi_w0, requirements='O'))
        else:
            coefs.append(np.require(w0, requirements='O'))

        n_iter[i] = n_iter_i

    if daal_ready:
        if fit_intercept:
            for i, ci in enumerate(coefs):
                coefs[i] = np.roll(ci, -1, -1)
        else:
            for i, ci in enumerate(coefs):
                coefs[i] = np.delete(ci, 0, axis=-1)

    if daal_ready:
        logging.info("sklearn.linear_model.LogisticRegression.fit: " + get_patch_message("daal"))
    else:
        logging.info("sklearn.linear_model.LogisticRegression.fit: " + get_patch_message("sklearn"))

    return coefs, np.array(Cs), n_iter


# Code adapted from sklearn.linear_model.logistic version 0.21
def __logistic_regression_path(X, y, pos_class=None, Cs=10, fit_intercept=True,
                              max_iter=100, tol=1e-4, verbose=0,
                              solver='lbfgs', coef=None,
                              class_weight=None, dual=False, penalty='l2',
                              intercept_scaling=1., multi_class='warn',
                              random_state=None, check_input=True,
                              max_squared_sum=None, sample_weight=None,
                              l1_ratio=None):
    """Compute a Logistic Regression model for a list of regularization
    parameters.

    This is an implementation that uses the result of the previous model
    to speed up computations along the set of solutions, making it faster
    than sequentially calling LogisticRegression for the different parameters.
    Note that there will be no speedup with liblinear solver, since it does
    not handle warm-starting.

    Read more in the :ref:`User Guide <logistic_regression>`.

    Parameters
    ----------
    X : array-like or sparse matrix, shape (n_samples, n_features)
        Input data.

    y : array-like, shape (n_samples,) or (n_samples, n_targets)
        Input data, target values.

    pos_class : int, None
        The class with respect to which we perform a one-vs-all fit.
        If None, then it is assumed that the given problem is binary.

    Cs : int | array-like, shape (n_cs,)
        List of values for the regularization parameter or integer specifying
        the number of regularization parameters that should be used. In this
        case, the parameters will be chosen in a logarithmic scale between
        1e-4 and 1e4.

    fit_intercept : bool
        Whether to fit an intercept for the model. In this case the shape of
        the returned array is (n_cs, n_features + 1).

    max_iter : int
        Maximum number of iterations for the solver.

    tol : float
        Stopping criterion. For the newton-cg and lbfgs solvers, the iteration
        will stop when ``max{|g_i | i = 1, ..., n} <= tol``
        where ``g_i`` is the i-th component of the gradient.

    verbose : int
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.

    solver : {'lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'}
        Numerical solver to use.

    coef : array-like, shape (n_features,), default None
        Initialization value for coefficients of logistic regression.
        Useless for liblinear solver.

    class_weight : dict or 'balanced', optional
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    dual : bool
        Dual or primal formulation. Dual formulation is only implemented for
        l2 penalty with liblinear solver. Prefer dual=False when
        n_samples > n_features.

    penalty : str, 'l1', 'l2', or 'elasticnet'
        Used to specify the norm used in the penalization. The 'newton-cg',
        'sag' and 'lbfgs' solvers support only l2 penalties. 'elasticnet' is
        only supported by the 'saga' solver.

    intercept_scaling : float, default 1.
        Useful only when the solver 'liblinear' is used
        and self.fit_intercept is set to True. In this case, x becomes
        [x, self.intercept_scaling],
        i.e. a "synthetic" feature with constant value equal to
        intercept_scaling is appended to the instance vector.
        The intercept becomes ``intercept_scaling * synthetic_feature_weight``.

        Note! the synthetic feature weight is subject to l1/l2 regularization
        as all other features.
        To lessen the effect of regularization on synthetic feature weight
        (and therefore on the intercept) intercept_scaling has to be increased.

    multi_class : str, {'ovr', 'multinomial', 'auto'}, default: 'ovr'
        If the option chosen is 'ovr', then a binary problem is fit for each
        label. For 'multinomial' the loss minimised is the multinomial loss fit
        across the entire probability distribution, *even when the data is
        binary*. 'multinomial' is unavailable when solver='liblinear'.
        'auto' selects 'ovr' if the data is binary, or if solver='liblinear',
        and otherwise selects 'multinomial'.

        .. versionadded:: 0.18
           Stochastic Average Gradient descent solver for 'multinomial' case.
        .. versionchanged:: 0.20
            Default will change from 'ovr' to 'auto' in 0.22.

    random_state : int, RandomState instance or None, optional, default None
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`. Used when ``solver`` == 'sag' or
        'liblinear'.

    check_input : bool, default True
        If False, the input arrays X and y will not be checked.

    max_squared_sum : float, default None
        Maximum squared sum of X over samples. Used only in SAG solver.
        If None, it will be computed, going through all the samples.
        The value should be precomputed to speed up cross validation.

    sample_weight : array-like, shape(n_samples,) optional
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight.

    l1_ratio : float or None, optional (default=None)
        The Elastic-Net mixing parameter, with ``0 <= l1_ratio <= 1``. Only
        used if ``penalty='elasticnet'``. Setting ``l1_ratio=0`` is equivalent
        to using ``penalty='l2'``, while setting ``l1_ratio=1`` is equivalent
        to using ``penalty='l1'``. For ``0 < l1_ratio <1``, the penalty is a
        combination of L1 and L2.

    Returns
    -------
    coefs : ndarray, shape (n_cs, n_features) or (n_cs, n_features + 1)
        List of coefficients for the Logistic Regression model. If
        fit_intercept is set to True then the second dimension will be
        n_features + 1, where the last item represents the intercept. For
        ``multiclass='multinomial'``, the shape is (n_classes, n_cs,
        n_features) or (n_classes, n_cs, n_features + 1).

    Cs : ndarray
        Grid of Cs used for cross-validation.

    n_iter : array, shape (n_cs,)
        Actual number of iteration for each Cs.

    Notes
    -----
    You might get slightly different results with the solver liblinear than
    with the others since this uses LIBLINEAR which penalizes the intercept.

    .. versionchanged:: 0.19
        The "copy" parameter was removed.
    """
    if isinstance(Cs, numbers.Integral):
        Cs = np.logspace(-4, 4, Cs)

    solver = _check_solver(solver, penalty, dual)

    # Preprocessing.
    if check_input:
        X = check_array(X, accept_sparse='csr', dtype=np.float64,
                        accept_large_sparse=solver != 'liblinear')
        y = check_array(y, ensure_2d=False, dtype=None)
        check_consistent_length(X, y)
    _, n_features = X.shape

    classes = np.unique(y)
    random_state = check_random_state(random_state)

    multi_class = _check_multi_class(multi_class, solver, len(classes))
    if pos_class is None and multi_class != 'multinomial':
        if (classes.size > 2):
            raise ValueError('To fit OvR, use the pos_class argument')
        # np.unique(y) gives labels in sorted order.
        pos_class = classes[1]

    # If sample weights exist, convert them to array (support for lists)
    # and check length
    # Otherwise set them to 1 for all examples
    if sample_weight is not None:
        default_weights = False
    else:
        default_weights = (class_weight is None)

    daal_ready = solver in ['lbfgs', 'newton-cg'] and not sparse.issparse(X)
    daal_ready = daal_ready and sample_weight is None and class_weight is None

    if not daal_ready:
        sample_weight = _check_sample_weight(sample_weight, X,
                                            dtype=X.dtype)
    # If class_weights is a dict (provided by the user), the weights
    # are assigned to the original labels. If it is "balanced", then
    # the class_weights are assigned after masking the labels with a OvR.
    le = LabelEncoder()
    if (isinstance(class_weight, dict) or multi_class == 'multinomial') and not daal_ready:
        class_weight_ = compute_class_weight(class_weight, classes=classes, y=y)
        if not np.allclose(class_weight_, np.ones_like(class_weight_)):
            sample_weight *= class_weight_[le.fit_transform(y)]

    # For doing a ovr, we need to mask the labels first. for the
    # multinomial case this is not necessary.
    if multi_class == 'ovr':
        w0 = np.zeros(n_features + int(fit_intercept), dtype=X.dtype)
        mask_classes = np.array([-1, 1])
        mask = (y == pos_class)
        y_bin = np.ones(y.shape, dtype=X.dtype)
        y_bin[~mask] = -1.
        # for compute_class_weight

        if class_weight == "balanced" and not daal_ready:
            class_weight_ = compute_class_weight(class_weight, classes=mask_classes,
                                                 y=y_bin)
            if not np.allclose(class_weight_, np.ones_like(class_weight_)):
                sample_weight *= class_weight_[le.fit_transform(y_bin)]

        if daal_ready:
            w0 = np.zeros(n_features + 1, dtype=X.dtype)
            y_bin[~mask] = 0.
        else:
            w0 = np.zeros(n_features + int(fit_intercept), dtype=X.dtype)

    else:
        if solver not in ['sag', 'saga']:
            if daal_ready:
                Y_multi = le.fit_transform(y).astype(X.dtype, copy=False)
            else:
                lbin = LabelBinarizer()
                Y_multi = lbin.fit_transform(y)
                if Y_multi.shape[1] == 1:
                    Y_multi = np.hstack([1 - Y_multi, Y_multi])
        else:
            # SAG multinomial solver needs LabelEncoder, not LabelBinarizer
            le = LabelEncoder()
            Y_multi = le.fit_transform(y).astype(X.dtype, copy=False)

        if daal_ready:
            w0 = np.zeros((classes.size, n_features + 1),
                      order='C', dtype=X.dtype)
        else:
            w0 = np.zeros((classes.size, n_features + int(fit_intercept)),
                      order='F', dtype=X.dtype)

    if coef is not None:
        # it must work both giving the bias term and not
        if multi_class == 'ovr':
            if coef.size not in (n_features, w0.size):
                raise ValueError(
                    'Initialization coef is of shape %d, expected shape '
                    '%d or %d' % (coef.size, n_features, w0.size))
            if daal_ready:
                w0[-coef.size:] = np.roll(coef, 1, -1) if coef.size != n_features else coef
            else:
                w0[:coef.size] = coef
        else:
            # For binary problems coef.shape[0] should be 1, otherwise it
            # should be classes.size.
            n_classes = classes.size
            if n_classes == 2:
                n_classes = 1

            if (coef.shape[0] != n_classes or
                    coef.shape[1] not in (n_features, n_features + 1)):
                raise ValueError(
                    'Initialization coef is of shape (%d, %d), expected '
                    'shape (%d, %d) or (%d, %d)' % (
                        coef.shape[0], coef.shape[1], classes.size,
                        n_features, classes.size, n_features + 1))

            if daal_ready:
                w0[:, -coef.shape[1]:] = np.roll(coef, 1, -1) if coef.shape[1] != n_features else coef
            else:
                if n_classes == 1:
                    w0[0, :coef.shape[1]] = -coef
                    w0[1, :coef.shape[1]] = coef
                else:
                    w0[:, :coef.shape[1]] = coef

    C_daal_multiplier = 1
    # commented out because this is Py3 feature
    #def _map_to_binary_logistic_regression():
    #    nonlocal C_daal_multiplier
    #    nonlocal w0
    #    C_daal_multiplier = 2
    #    w0 *= 2

    if multi_class == 'multinomial':
        # fmin_l_bfgs_b and newton-cg accepts only ravelled parameters.
        if solver in ['lbfgs', 'newton-cg']:
            if daal_ready and classes.size == 2:
                w0_saved = w0
                w0 = w0[-1:, :]
            w0 = w0.ravel()
        target = Y_multi
        if solver == 'lbfgs':
            if daal_ready:
                if classes.size == 2:
                    # _map_to_binary_logistic_regression()
                    C_daal_multiplier = 2
                    w0 *= 2
                    daal_extra_args_func = _daal4py_logistic_loss_extra_args
                else:
                    daal_extra_args_func = _daal4py_cross_entropy_loss_extra_args
                func = _daal4py_loss_and_grad
            else:
                func = lambda x, *args: _multinomial_loss_grad(x, *args)[0:2]
        elif solver == 'newton-cg':
            if daal_ready:
                if classes.size == 2:
                    # _map_to_binary_logistic_regression()
                    C_daal_multiplier = 2
                    w0 *= 2
                    daal_extra_args_func = _daal4py_logistic_loss_extra_args
                else:
                    daal_extra_args_func = _daal4py_cross_entropy_loss_extra_args
                func = _daal4py_loss_
                grad = _daal4py_grad_
                hess = _daal4py_grad_hess_
            else:
                func = lambda x, *args: _multinomial_loss(x, *args)[0]
                grad = lambda x, *args: _multinomial_loss_grad(x, *args)[1]
                hess = _multinomial_grad_hess
        warm_start_sag = {'coef': w0.T}
    else:
        target = y_bin
        if solver == 'lbfgs':
            if daal_ready:
                func = _daal4py_loss_and_grad
                daal_extra_args_func = _daal4py_logistic_loss_extra_args
            else:
                func = _logistic_loss_and_grad
        elif solver == 'newton-cg':
            if daal_ready:
                daal_extra_args_func = _daal4py_logistic_loss_extra_args
                func = _daal4py_loss_
                grad = _daal4py_grad_
                hess = _daal4py_grad_hess_
            else:
                func = _logistic_loss
                grad = lambda x, *args: _logistic_loss_and_grad(x, *args)[1]
                hess = _logistic_grad_hess
        warm_start_sag = {'coef': np.expand_dims(w0, axis=1)}

    coefs = list()
    n_iter = np.zeros(len(Cs), dtype=np.int32)
    for i, C in enumerate(Cs):
        if solver == 'lbfgs':
            if daal_ready:
                extra_args = daal_extra_args_func(classes.size, w0, X, target, 0., 0.5 / C / C_daal_multiplier,
                                                  fit_intercept, value=True, gradient=True, hessian=False)
            else:
                extra_args = (X, target, 1. / C, sample_weight)

            iprint = [-1, 50, 1, 100, 101][
                np.searchsorted(np.array([0, 1, 2, 3]), verbose)]
            opt_res = optimize.minimize(
                func, w0, method="L-BFGS-B", jac=True,
                args=extra_args,
                options={"iprint": iprint, "gtol": tol, "maxiter": max_iter}
            )
            n_iter_i = _check_optimize_result(
                solver, opt_res, max_iter,
                extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
            w0, loss = opt_res.x, opt_res.fun
            if daal_ready and C_daal_multiplier == 2:
                w0 *= 0.5
        elif solver == 'newton-cg':
            if daal_ready:
                def make_ncg_funcs(f, value=False, gradient=False, hessian=False):
                    daal_penaltyL2 = 0.5 / C / C_daal_multiplier
                    _obj_, X_, y_, n_samples = daal_extra_args_func(
                        classes.size, w0, X, target, 0., daal_penaltyL2, fit_intercept,
                        value=value, gradient=gradient, hessian=hessian)
                    _func_ = lambda x, *args: f(x, _obj_, *args)
                    return _func_, (X_, y_, n_samples, daal_penaltyL2)

                loss_func, extra_args  = make_ncg_funcs(func, value=True)
                grad_func, _  = make_ncg_funcs(grad, gradient=True)
                grad_hess_func, _ = make_ncg_funcs(hess, gradient=True)
                w0, n_iter_i = _newton_cg(grad_hess_func, loss_func, grad_func, w0, args=extra_args,
                                         maxiter=max_iter, tol=tol)
            else:
                args = (X, target, 1. / C, sample_weight)
                w0, n_iter_i = _newton_cg(hess, func, grad, w0, args=args,
                                         maxiter=max_iter, tol=tol)
        elif solver == 'liblinear':
            coef_, intercept_, n_iter_i, = _fit_liblinear(
                X, target, C, fit_intercept, intercept_scaling, None,
                penalty, dual, verbose, max_iter, tol, random_state,
                sample_weight=sample_weight)
            if fit_intercept:
                w0 = np.concatenate([coef_.ravel(), intercept_])
            else:
                w0 = coef_.ravel()

        elif solver in ['sag', 'saga']:
            if multi_class == 'multinomial':
                target = target.astype(X.dtype, copy=False)
                loss = 'multinomial'
            else:
                loss = 'log'
            # alpha is for L2-norm, beta is for L1-norm
            if penalty == 'l1':
                alpha = 0.
                beta = 1. / C
            elif penalty == 'l2':
                alpha = 1. / C
                beta = 0.
            else:  # Elastic-Net penalty
                alpha = (1. / C) * (1 - l1_ratio)
                beta = (1. / C) * l1_ratio

            w0, n_iter_i, warm_start_sag = sag_solver(
                X, target, sample_weight, loss, alpha,
                beta, max_iter, tol,
                verbose, random_state, False, max_squared_sum, warm_start_sag,
                is_saga=(solver == 'saga'))

        else:
            raise ValueError("solver must be one of {'liblinear', 'lbfgs', "
                             "'newton-cg', 'sag'}, got '%s' instead" % solver)

        if multi_class == 'multinomial':
            if daal_ready:
                if classes.size == 2:
                   multi_w0 = w0[np.newaxis, :]
                else:
                   multi_w0 = np.reshape(w0, (classes.size, -1))
            else:
                n_classes = max(2, classes.size)
                multi_w0 = np.reshape(w0, (n_classes, -1))
                if n_classes == 2:
                    multi_w0 = multi_w0[1][np.newaxis, :]
            coefs.append(np.require(multi_w0, requirements='O'))
        else:
            coefs.append(np.require(w0, requirements='O'))

        n_iter[i] = n_iter_i

    if daal_ready:
        if fit_intercept:
            for i, ci in enumerate(coefs):
                coefs[i] = np.roll(ci, -1, -1)
        else:
            for i, ci in enumerate(coefs):
                coefs[i] = np.delete(ci, 0, axis=-1)

    if daal_ready:
        logging.info("sklearn.linear_model.LogisticRegression.fit: " + get_patch_message("daal"))
    else:
        logging.info("sklearn.linear_model.LogisticRegression.fit: " + get_patch_message("sklearn"))

    return np.array(coefs), np.array(Cs), n_iter


def daal4py_predict(self, X, resultsToEvaluate):
    check_is_fitted(self)
    X = check_array(X, accept_sparse='csr', dtype=[np.float64, np.float32])
    try:
        fptype = getFPType(X)
    except ValueError:
        fptype = None

    multinomial = (self.multi_class in ["multinomial", "warn"] or
                   self.classes_.size == 2 or resultsToEvaluate == 'computeClassLabels')

    if daal_check_version(((2021,'P', 1))) and fptype is not None and not sparse.issparse(X) \
       and multinomial and not sparse.issparse(self.coef_):
        logging.info("sklearn.linear_model.LogisticRegression.predict: " + get_patch_message("daal"))
        n_features = self.coef_.shape[1]
        if X.shape[1] != n_features:
            raise ValueError(f'X has {X.shape[1]} features, but expecting {n_features} features per sample')
        builder = d4p.logistic_regression_model_builder(X.shape[1], len(self.classes_))
        builder.set_beta(self.coef_, self.intercept_)
        predict = d4p.logistic_regression_prediction(nClasses=len(self.classes_),
                                                    fptype=fptype,
                                                    method = 'defaultDense',
                                                    resultsToEvaluate = resultsToEvaluate)
        res = predict.compute(X, builder.model)
        if resultsToEvaluate == 'computeClassLabels':
            res = res.prediction
            if not np.array_equal(self.classes_, np.arange(0, len(self.classes_))) or \
            self.classes_.dtype != X.dtype:
                res = self.classes_.take(np.asarray(res, dtype=np.intp))
        elif resultsToEvaluate == 'computeClassProbabilities':
            res = res.probabilities
        elif resultsToEvaluate == 'computeClassLogProbabilities':
            res = res.logProbabilities
        else:
            raise ValueError('resultsToEvaluate must be in [computeClassLabels, \
                             computeClassProbabilities, computeClassLogProbabilities]')
        if res.shape[1] == 1:
            res = np.ravel(res)
        return res

    if resultsToEvaluate == 'computeClassLabels':
        logging.info("sklearn.linear_model.LogisticRegression.predict: " + get_patch_message("sklearn"))
        return LogisticRegression_original.predict(self, X)
    if resultsToEvaluate == 'computeClassProbabilities':
        logging.info("sklearn.linear_model.LogisticRegression.predict_proba: " + get_patch_message("sklearn"))
        return LogisticRegression_original.predict_proba(self, X)
    if resultsToEvaluate == 'computeClassLogProbabilities':
        logging.info("sklearn.linear_model.LogisticRegression.predict_log_proba: " + get_patch_message("sklearn"))
        return LogisticRegression_original.predict_log_proba(self, X)
    raise ValueError('resultsToEvaluate must be in [computeClassLabels, \
                     computeClassProbabilities, computeClassLogProbabilities]')


if (LooseVersion(sklearn_version) >= LooseVersion("0.24")):
    def _logistic_regression_path(X, y, pos_class=None, Cs=10, fit_intercept=True,
                              max_iter=100, tol=1e-4, verbose=0,
                              solver='lbfgs', coef=None,
                              class_weight=None, dual=False, penalty='l2',
                              intercept_scaling=1., multi_class='auto',
                              random_state=None, check_input=True,
                              max_squared_sum=None, sample_weight=None,
                              l1_ratio=None):
        return __logistic_regression_path(X, y, pos_class=pos_class,
                                          Cs=Cs, fit_intercept=fit_intercept,
                                          max_iter=max_iter, tol=tol, verbose=verbose,
                                          solver=solver, coef=coef,
                                          class_weight=class_weight,
                                          dual=dual, penalty=penalty,
                                          intercept_scaling=intercept_scaling,
                                          multi_class=multi_class,
                                          random_state=random_state,
                                          check_input=check_input,
                                          max_squared_sum=max_squared_sum,
                                          sample_weight=sample_weight,
                                          l1_ratio=l1_ratio)
    class LogisticRegression(LogisticRegression_original, LinearClassifierMixin,
                             SparseCoefMixin, BaseEstimator):
        def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0,
                     fit_intercept=True, intercept_scaling=1, class_weight=None,
                     random_state=None, solver='lbfgs', max_iter=100,
                     multi_class='auto', verbose=0, warm_start=False, n_jobs=None,
                     l1_ratio=None):
            self.penalty = penalty
            self.dual = dual
            self.tol = tol
            self.C = C
            self.fit_intercept = fit_intercept
            self.intercept_scaling = intercept_scaling
            self.class_weight = class_weight
            self.random_state = random_state
            self.solver = solver
            self.max_iter = max_iter
            self.multi_class = multi_class
            self.verbose = verbose
            self.warm_start = warm_start
            self.n_jobs = 1 if n_jobs is not None else None
            self.l1_ratio = l1_ratio


        def predict(self, X):
            return daal4py_predict(self, X, 'computeClassLabels')


        def predict_log_proba(self, X):
            return daal4py_predict(self, X, 'computeClassLogProbabilities')


        def predict_proba(self, X):
            return daal4py_predict(self, X, 'computeClassProbabilities')


elif (LooseVersion(sklearn_version) >= LooseVersion("0.22")):
    def _logistic_regression_path(X, y, pos_class=None, Cs=10, fit_intercept=True,
                              max_iter=100, tol=1e-4, verbose=0,
                              solver='lbfgs', coef=None,
                              class_weight=None, dual=False, penalty='l2',
                              intercept_scaling=1., multi_class='auto',
                              random_state=None, check_input=True,
                              max_squared_sum=None, sample_weight=None,
                              l1_ratio=None):
        return __logistic_regression_path(X, y, pos_class=pos_class,
                                          Cs=Cs, fit_intercept=fit_intercept,
                                          max_iter=max_iter, tol=tol, verbose=verbose,
                                          solver=solver, coef=coef,
                                          class_weight=class_weight,
                                          dual=dual, penalty=penalty,
                                          intercept_scaling=intercept_scaling,
                                          multi_class=multi_class,
                                          random_state=random_state,
                                          check_input=check_input,
                                          max_squared_sum=max_squared_sum,
                                          sample_weight=sample_weight,
                                          l1_ratio=l1_ratio)
    class LogisticRegression(LogisticRegression_original, BaseEstimator,
                             LinearClassifierMixin, SparseCoefMixin):
        def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0,
                     fit_intercept=True, intercept_scaling=1, class_weight=None,
                     random_state=None, solver='lbfgs', max_iter=100,
                     multi_class='auto', verbose=0, warm_start=False, n_jobs=None,
                     l1_ratio=None):

            self.penalty = penalty
            self.dual = dual
            self.tol = tol
            self.C = C
            self.fit_intercept = fit_intercept
            self.intercept_scaling = intercept_scaling
            self.class_weight = class_weight
            self.random_state = random_state
            self.solver = solver
            self.max_iter = max_iter
            self.multi_class = multi_class
            self.verbose = verbose
            self.warm_start = warm_start
            self.n_jobs = 1 if n_jobs is not None else None
            self.l1_ratio = l1_ratio

        def predict(self, X):
            return daal4py_predict(self, X, 'computeClassLabels')


        def predict_log_proba(self, X):
            return daal4py_predict(self, X, 'computeClassLogProbabilities')


        def predict_proba(self, X):
            return daal4py_predict(self, X, 'computeClassProbabilities')

elif (LooseVersion(sklearn_version) >= LooseVersion("0.21")):
    def _logistic_regression_path(X, y, pos_class=None, Cs=10, fit_intercept=True,
                              max_iter=100, tol=1e-4, verbose=0,
                              solver='lbfgs', coef=None,
                              class_weight=None, dual=False, penalty='l2',
                              intercept_scaling=1., multi_class='warn',
                              random_state=None, check_input=True,
                              max_squared_sum=None, sample_weight=None,
                              l1_ratio=None):
        return __logistic_regression_path(X, y, pos_class=pos_class,
                                          Cs=Cs, fit_intercept=fit_intercept,
                                          max_iter=max_iter, tol=tol, verbose=verbose,
                                          solver=solver, coef=coef,
                                          class_weight=class_weight,
                                          dual=dual, penalty=penalty,
                                          intercept_scaling=intercept_scaling,
                                          multi_class=multi_class,
                                          random_state=random_state,
                                          check_input=check_input,
                                          max_squared_sum=max_squared_sum,
                                          sample_weight=sample_weight,
                                          l1_ratio=l1_ratio)
    class LogisticRegression(LogisticRegression_original, BaseEstimator,
                             LinearClassifierMixin, SparseCoefMixin):
        def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0,
                     fit_intercept=True, intercept_scaling=1, class_weight=None,
                     random_state=None, solver='warn', max_iter=100,
                     multi_class='warn', verbose=0, warm_start=False, n_jobs=None,
                     l1_ratio=None):

            self.penalty = penalty
            self.dual = dual
            self.tol = tol
            self.C = C
            self.fit_intercept = fit_intercept
            self.intercept_scaling = intercept_scaling
            self.class_weight = class_weight
            self.random_state = random_state
            self.solver = solver
            self.max_iter = max_iter
            self.multi_class = multi_class
            self.verbose = verbose
            self.warm_start = warm_start
            self.n_jobs = 1 if n_jobs is not None else None
            self.l1_ratio = l1_ratio


        def predict(self, X):
            return daal4py_predict(self, X, 'computeClassLabels')


        def predict_log_proba(self, X):
            return daal4py_predict(self, X, 'computeClassLogProbabilities')


        def predict_proba(self, X):
            return daal4py_predict(self, X, 'computeClassProbabilities')

else:
    class LogisticRegression(LogisticRegression_original, BaseEstimator,
                             LinearClassifierMixin, SparseCoefMixin):
        def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0,
                     fit_intercept=True, intercept_scaling=1, class_weight=None,
                     random_state=None, solver='warn', max_iter=100,
                     multi_class='warn', verbose=0, warm_start=False, n_jobs=None):

            self.penalty = penalty
            self.dual = dual
            self.tol = tol
            self.C = C
            self.fit_intercept = fit_intercept
            self.intercept_scaling = intercept_scaling
            self.class_weight = class_weight
            self.random_state = random_state
            self.solver = solver
            self.max_iter = max_iter
            self.multi_class = multi_class
            self.verbose = verbose
            self.warm_start = warm_start
            self.n_jobs = 1 if n_jobs is not None else None

        def predict(self, X):
            return daal4py_predict(self, X, 'computeClassLabels')


        def predict_log_proba(self, X):
            return daal4py_predict(self, X, 'computeClassLogProbabilities')


        def predict_proba(self, X):
            return daal4py_predict(self, X, 'computeClassProbabilities')
