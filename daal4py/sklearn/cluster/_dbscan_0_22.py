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

import numpy as np
import warnings
from scipy import sparse

from sklearn.utils import check_array
from sklearn.utils.validation import _check_sample_weight

from sklearn.cluster import DBSCAN as DBSCAN_original

import daal4py
from daal4py.sklearn._utils import (make2d, getFPType, get_patch_message)
import logging


def _daal_dbscan(X, eps=0.5, min_samples=5, sample_weight=None):
    if eps <= 0.0:
        raise ValueError("eps must be positive.")

    X = check_array(X, dtype=[np.float64, np.float32])
    if sample_weight is not None:
        sample_weight = _check_sample_weight(sample_weight, X)
        ww = make2d(sample_weight)
    else:
        ww = None

    XX = make2d(X)

    fpt = getFPType(XX)
    alg = daal4py.dbscan(
        method='defaultDense',
        fptype = fpt,
        epsilon=float(eps),
        minObservations=int(min_samples),
        memorySavingMode=False,
        resultsToCompute="computeCoreIndices")

    daal_res = alg.compute(XX, ww)
    n_clusters = daal_res.nClusters[0, 0]
    assignments = daal_res.assignments.ravel()
    if daal_res.coreIndices is not None:
        core_ind = daal_res.coreIndices.ravel()
    else:
        core_ind = np.array([], dtype=np.intc)

    return (core_ind, assignments)


class DBSCAN(DBSCAN_original):
    """Perform DBSCAN clustering from vector array or distance matrix.

    DBSCAN - Density-Based Spatial Clustering of Applications with Noise.
    Finds core samples of high density and expands clusters from them.
    Good for data which contains clusters of similar density.

    Read more in the :ref:`User Guide <dbscan>`.

    Parameters
    ----------
    eps : float, optional
        The maximum distance between two samples for one to be considered
        as in the neighborhood of the other. This is not a maximum bound
        on the distances of points within a cluster. This is the most
        important DBSCAN parameter to choose appropriately for your data set
        and distance function.

    min_samples : int, optional
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.

    metric : string, or callable
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of
        the options allowed by :func:`sklearn.metrics.pairwise_distances` for
        its metric parameter.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square. X may be a :term:`Glossary <sparse graph>`, in which
        case only "nonzero" elements may be considered neighbors for DBSCAN.

        .. versionadded:: 0.17
           metric *precomputed* to accept precomputed sparse matrix.

    metric_params : dict, optional
        Additional keyword arguments for the metric function.

        .. versionadded:: 0.19

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute', 'daal'}, optional
        The algorithm to be used by the NearestNeighbors module
        to compute pointwise distances and find nearest neighbors.
        See NearestNeighbors module documentation for details.

        If algorithm is set to 'daal', Intel(R) oneAPI Data Analytics Library
        will be used.

    leaf_size : int, optional (default = 30)
        Leaf size passed to BallTree or cKDTree. This can affect the speed
        of the construction and query, as well as the memory required
        to store the tree. The optimal value depends
        on the nature of the problem.

    p : float, optional
        The power of the Minkowski metric to be used to calculate distance
        between points.

    n_jobs : int or None, optional (default=None)
        The number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    core_sample_indices_ : array, shape = [n_core_samples]
        Indices of core samples.

    components_ : array, shape = [n_core_samples, n_features]
        Copy of each core sample found by training.

    labels_ : array, shape = [n_samples]
        Cluster labels for each point in the dataset given to fit().
        Noisy samples are given the label -1.

    Examples
    --------
    >>> from sklearn.cluster import DBSCAN
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 2], [2, 3],
    ...               [8, 7], [8, 8], [25, 80]])
    >>> clustering = DBSCAN(eps=3, min_samples=2).fit(X)
    >>> clustering.labels_
    array([ 0,  0,  0,  1,  1, -1])
    >>> clustering
    DBSCAN(eps=3, min_samples=2)

    See also
    --------
    OPTICS
        A similar clustering at multiple values of eps. Our implementation
        is optimized for memory usage.

    Notes
    -----
    For an example, see :ref:`examples/cluster/plot_dbscan.py
    <sphx_glr_auto_examples_cluster_plot_dbscan.py>`.

    This implementation bulk-computes all neighborhood queries, which increases
    the memory complexity to O(n.d) where d is the average number of neighbors,
    while original DBSCAN had memory complexity O(n). It may attract a higher
    memory complexity when querying these nearest neighborhoods, depending
    on the ``algorithm``.

    One way to avoid the query complexity is to pre-compute sparse
    neighborhoods in chunks using
    :func:`NearestNeighbors.radius_neighbors_graph
    <sklearn.neighbors.NearestNeighbors.radius_neighbors_graph>` with
    ``mode='distance'``, then using ``metric='precomputed'`` here.

    Another way to reduce memory and computation time is to remove
    (near-)duplicate points and use ``sample_weight`` instead.

    :class:`cluster.OPTICS` provides a similar clustering with lower memory
    usage.

    References
    ----------
    Ester, M., H. P. Kriegel, J. Sander, and X. Xu, "A Density-Based
    Algorithm for Discovering Clusters in Large Spatial Databases with Noise".
    In: Proceedings of the 2nd International Conference on Knowledge Discovery
    and Data Mining, Portland, OR, AAAI Press, pp. 226-231. 1996

    Schubert, E., Sander, J., Ester, M., Kriegel, H. P., & Xu, X. (2017).
    DBSCAN revisited, revisited: why and how you should (still) use DBSCAN.
    ACM Transactions on Database Systems (TODS), 42(3), 19.
    """

    def __init__(self, eps=0.5, min_samples=5, metric='euclidean',
                 metric_params=None, algorithm='auto', leaf_size=30, p=None,
                 n_jobs=None):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.metric_params = metric_params
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.n_jobs = n_jobs

    def fit(self, X, y=None, sample_weight=None):
        """Perform DBSCAN clustering from features, or distance matrix.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features), or \
            (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``metric='precomputed'``. If a sparse matrix is provided, it will
            be converted into a sparse ``csr_matrix``.

        sample_weight : array, shape (n_samples,), optional
            Weight of each sample, such that a sample with a weight of at least
            ``min_samples`` is by itself a core sample; a sample with a
            negative weight may inhibit its eps-neighbor from being core.
            Note that weights are absolute, and default to 1.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self

        """
        X = check_array(X, accept_sparse='csr', dtype=[np.float64, np.float32])

        if self.eps <= 0.0:
            raise ValueError("eps must be positive.")

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        _daal_ready = ((self.algorithm in ['auto', 'brute']) and
                       (self.metric == 'euclidean' or
                       (self.metric == 'minkowski' and self.p == 2)) and 
                       isinstance(X, np.ndarray))
        if _daal_ready:
            logging.info("sklearn.cluster.DBSCAN.fit: " + get_patch_message("daal"))
            core_ind, assignments = _daal_dbscan(
                X, self.eps,
                self.min_samples,
                sample_weight=sample_weight)
            self.core_sample_indices_ = core_ind
            self.labels_ = assignments
            self.components_ = np.take(X, core_ind, axis=0)
            return self
        logging.info("sklearn.cluster.DBSCAN.fit: " + get_patch_message("sklearn"))
        return super().fit(X, y, sample_weight=sample_weight)
