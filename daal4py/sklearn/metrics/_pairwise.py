#===============================================================================
# Copyright 2014 Intel Corporation
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

import numpy as np
from functools import partial
from sklearn.metrics.pairwise import _parallel_pairwise, _pairwise_callable
from sklearn.metrics.pairwise import _VALID_METRICS, PAIRWISE_DISTANCE_FUNCTIONS
from sklearn.metrics.pairwise import PAIRWISE_BOOLEAN_FUNCTIONS
from sklearn.metrics.pairwise import check_pairwise_arrays
from sklearn.utils._joblib import effective_n_jobs
from sklearn.utils.validation import check_non_negative
import warnings
from sklearn.exceptions import DataConversionWarning
try:
    from sklearn.metrics.pairwise import _precompute_metric_params
except ImportError:
    def _precompute_metric_params(*args, **kwrds):
        return dict()

from scipy.sparse import issparse
from scipy.spatial import distance

import daal4py
from daal4py.sklearn.utils.validation import _daal_check_array
from .._utils import (getFPType, PatchingConditionsChain)
from .._device_offload import support_usm_ndarray


def _daal4py_cosine_distance_dense(X):
    X_fptype = getFPType(X)
    alg = daal4py.cosine_distance(fptype=X_fptype, method='defaultDense')
    res = alg.compute(X)
    return res.cosineDistance


def _daal4py_correlation_distance_dense(X):
    X_fptype = getFPType(X)
    alg = daal4py.correlation_distance(fptype=X_fptype, method='defaultDense')
    res = alg.compute(X)
    return res.correlationDistance


@support_usm_ndarray(freefunc=True)
def daal_pairwise_distances(X, Y=None, metric="euclidean", n_jobs=None,
                            force_all_finite=True, **kwds):
    """ Compute the distance matrix from a vector array X and optional Y.

    This method takes either a vector array or a distance matrix, and returns
    a distance matrix. If the input is a vector array, the distances are
    computed. If the input is a distances matrix, it is returned instead.

    This method provides a safe way to take a distance matrix as input, while
    preserving compatibility with many other algorithms that take a vector
    array.

    If Y is given (default is None), then the returned matrix is the pairwise
    distance between the arrays from both X and Y.

    Valid values for metric are:

    - From scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
      'manhattan']. These metrics support sparse matrix
      inputs.
      ['nan_euclidean'] but it does not yet support sparse matrices.

    - From scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
      'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
      'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
      'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
      See the documentation for scipy.spatial.distance for details on these
      metrics. These metrics do not support sparse matrix inputs.

    Note that in the case of 'cityblock', 'cosine' and 'euclidean' (which are
    valid scipy.spatial.distance metrics), the scikit-learn implementation
    will be used, which is faster and has support for sparse matrices (except
    for 'cityblock'). For a verbose description of the metrics from
    scikit-learn, see the __doc__ of the sklearn.pairwise.distance_metrics
    function.

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : array [n_samples_a, n_samples_a] if metric == "precomputed", or, \
             [n_samples_a, n_features] otherwise
        Array of pairwise distances between samples, or a feature array.

    Y : array [n_samples_b, n_features], optional
        An optional second feature array. Only allowed if
        metric != "precomputed".

    metric : string, or callable
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by scipy.spatial.distance.pdist for its metric parameter, or
        a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
        If metric is "precomputed", X is assumed to be a distance matrix.
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays from X as input and return a value indicating
        the distance between them.

    n_jobs : int or None, optional (default=None)
        The number of jobs to use for the computation. This works by breaking
        down the pairwise matrix into n_jobs even slices and computing them in
        parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    force_all_finite : boolean or 'allow-nan', (default=True)
        Whether to raise an error on np.inf and np.nan in array. The
        possibilities are:

        - True: Force all values of array to be finite.
        - False: accept both np.inf and np.nan in array.
        - 'allow-nan': accept only np.nan values in array. Values cannot
          be infinite.

        .. versionadded:: 0.22

    **kwds : optional keyword parameters
        Any further parameters are passed directly to the distance function.
        If using a scipy.spatial.distance metric, the parameters are still
        metric dependent. See the scipy docs for usage examples.

    Returns
    -------
    D : array [n_samples_a, n_samples_a] or [n_samples_a, n_samples_b]
        A distance matrix D such that D_{i, j} is the distance between the
        ith and jth vectors of the given matrix X, if Y is None.
        If Y is not None, then D_{i, j} is the distance between the ith array
        from X and the jth array from Y.

    See also
    --------
    pairwise_distances_chunked : performs the same calculation as this
        function, but returns a generator of chunks of the distance matrix, in
        order to limit memory usage.
    paired_distances : Computes the distances between corresponding
                       elements of two arrays
    """
    if metric not in _VALID_METRICS and not callable(metric) and metric != "precomputed":
        raise ValueError("Unknown metric %s. Valid metrics are %s, or 'precomputed', "
                         "or a callable" % (metric, _VALID_METRICS))

    X = _daal_check_array(X, accept_sparse=['csr', 'csc', 'coo'],
                          force_all_finite=force_all_finite)

    _patching_status = PatchingConditionsChain(
        "sklearn.metrics.pairwise_distances")
    _dal_ready = _patching_status.and_conditions([
        (metric == 'cosine' or metric == 'correlation',
            f"'{metric}' metric is not supported. "
            "Only 'cosine' and 'correlation' metrics are supported."),
        (Y is None, "Second feature array is not supported."),
        (not issparse(X), "X is sparse. Sparse input is not supported."),
        (X.dtype == np.float64,
            f"{X.dtype} X data type is not supported. Only np.float64 is supported.")
    ])
    _patching_status.write_log()
    if _dal_ready:
        if metric == 'cosine':
            return _daal4py_cosine_distance_dense(X)
        if metric == 'correlation':
            return _daal4py_correlation_distance_dense(X)
        raise ValueError(f"'{metric}' distance is wrong for daal4py.")
    if metric == "precomputed":
        X, _ = check_pairwise_arrays(X, Y, precomputed=True,
                                     force_all_finite=force_all_finite)
        whom = ("`pairwise_distances`. Precomputed distance "
                " need to have non-negative values.")
        check_non_negative(X, whom=whom)
        return X
    if metric in PAIRWISE_DISTANCE_FUNCTIONS:
        func = PAIRWISE_DISTANCE_FUNCTIONS[metric]
    elif callable(metric):
        func = partial(_pairwise_callable, metric=metric,
                       force_all_finite=force_all_finite, **kwds)
    else:
        if issparse(X) or issparse(Y):
            raise TypeError("scipy distance metrics do not"
                            " support sparse matrices.")

        dtype = bool if metric in PAIRWISE_BOOLEAN_FUNCTIONS else None

        if dtype == bool and (X.dtype != bool or (Y is not None and Y.dtype != bool)):
            msg = "Data was converted to boolean for metric %s" % metric
            warnings.warn(msg, DataConversionWarning)

        X, Y = check_pairwise_arrays(X, Y, dtype=dtype,
                                     force_all_finite=force_all_finite)

        # precompute data-derived metric params
        params = _precompute_metric_params(X, Y, metric=metric, **kwds)
        kwds.update(**params)

        if effective_n_jobs(n_jobs) == 1 and X is Y:
            return distance.squareform(distance.pdist(X, metric=metric,
                                                      **kwds))
        func = partial(distance.cdist, metric=metric, **kwds)

    return _parallel_pairwise(X, Y, func, n_jobs, **kwds)
