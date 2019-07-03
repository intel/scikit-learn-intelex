"""
Tests for DBSCAN clustering algorithm
"""

import pickle
import pytest
import numpy as np

from scipy.spatial import distance
from scipy import sparse

from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_in
from sklearn.utils.testing import assert_not_in
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster.dbscan_ import DBSCAN as DBSCAN_SKLEARN
from sklearn.cluster.dbscan_ import dbscan as dbscan_sklearn

from wrappers.wrapper_dbscan import DBSCAN as DBSCAN_DAAL
from wrappers.wrapper_dbscan import dbscan as dbscan_daal

from sklearn.cluster.tests.common import generate_clustered_data
from sklearn.metrics.pairwise import pairwise_distances

n_clusters = 3
X = generate_clustered_data(n_clusters=n_clusters)

def generate_data(low: int, high: int, samples_number: int, sample_dimension: tuple) -> tuple:
    generator = np.random.RandomState()
    table_size = (samples_number, sample_dimension)
    return generator.uniform(low=low, high=high, size=table_size), generator.uniform(size=samples_number)

@pytest.fixture(params=[False])
def use_sparse(request):
    return request.param

@pytest.fixture(params=[True, False])
def use_weights(request):
    return request.param

@pytest.fixture(params=["euclidean"])
def metric(request):
    return request.param

@pytest.fixture(params=[("euclidean", 1)])
def metric_with_param(request):
    return request.param

@pytest.fixture(params=["default"])
def algorithm(request):
    return request.param

@pytest.mark.skip(reason="No functionality for precomputed data processing")
def test_dbscan_similarity():
    # Tests the DBSCAN algorithm with a similarity array.
    # Parameters chosen specifically for this task.
    eps = 0.15
    min_samples = 10
    # Compute similarities
    D = distance.squareform(distance.pdist(X))
    D /= np.max(D)
    # Compute DBSCAN
    _, labels = dbscan_sklearn(D, metric="precomputed", eps=eps,
                                  min_samples=min_samples)
    # number of clusters, ignoring noise if present
    n_clusters_1 = len(set(labels)) - (1 if -1 in labels else 0)

    assert_equal(n_clusters_1, n_clusters)

    db = DBSCAN_DAAL(metric="precomputed", eps=eps, min_samples=min_samples)
    labels = db.fit(D).labels_

    n_clusters_2 = len(set(labels)) - int(-1 in labels)
    assert_equal(n_clusters_2, n_clusters)

def test_dbscan_feature(metric):
    # Tests the DBSCAN algorithm with a feature vector array.
    # Parameters chosen specifically for this task.
    # Different eps to other test, because distance is not normalised.
    eps = 0.8
    min_samples = 10
    # Compute DBSCAN
    # parameters chosen for task
    _, labels = dbscan_sklearn(X, metric=metric, eps=eps,
                                min_samples=min_samples)
    # number of clusters, ignoring noise if present
    n_clusters_1 = len(set(labels)) - int(-1 in labels)
    assert_equal(n_clusters_1, n_clusters)

    db = DBSCAN_DAAL(metric=metric, eps=eps, min_samples=min_samples)
    labels = db.fit(X).labels_

    n_clusters_2 = len(set(labels)) - int(-1 in labels)
    assert_equal(n_clusters_2, n_clusters)

@pytest.mark.skip(reason="No functionality for sparse matrix processing")
def test_dbscan_sparse():
    core_sparse, labels_sparse = dbscan_daal(sparse.lil_matrix(X), eps=.8,
                                        min_samples=10)
    core_dense, labels_dense = dbscan_daal(X, eps=.8, min_samples=10)
    assert_array_equal(core_dense, core_sparse)
    assert_array_equal(labels_dense, labels_sparse)

@pytest.mark.skip(reason="No functionality for sparse matrix and precomputed data processing")
@pytest.mark.parametrize('include_self', [False, True])
def test_dbscan_sparse_precomputed(include_self):
    D = pairwise_distances(X)
    nn = NearestNeighbors(radius=.9).fit(X)
    X_ = X if include_self else None
    D_sparse = nn.radius_neighbors_graph(X=X_, mode='distance')
    # Ensure it is sparse not merely on diagonals:
    assert D_sparse.nnz < D.shape[0] * (D.shape[0] - 1)
    core_sparse, labels_sparse = dbscan_daal(D_sparse,
                                        eps=.8,
                                        min_samples=10,
                                        metric='precomputed')
    core_dense, labels_dense = dbscan_daal(D, eps=.8, min_samples=10,
                                      metric='precomputed')
    assert_array_equal(core_dense, core_sparse)
    assert_array_equal(labels_dense, labels_sparse)

def test_dbscan_input_not_modified(use_sparse, metric):
    # test that the input is not modified by dbscan
    X = np.random.RandomState(0).rand(10, 10)
    X = sparse.csr_matrix(X) if use_sparse else X
    X_copy = X.copy()
    dbscan_daal(X, metric=metric)

    if use_sparse:
        assert_array_equal(X.toarray(), X_copy.toarray())
    else:
        assert_array_equal(X, X_copy)

def test_dbscan_no_core_samples(use_sparse):
    rng = np.random.RandomState(0)
    X = rng.rand(40, 10)
    X[X < .8] = 0
    if use_sparse:
        X = sparse.csr_matrix(X)
    db = DBSCAN_DAAL(min_samples=6).fit(X)
    assert_array_equal(db.components_, np.empty((0, X.shape[1])))
    assert_array_equal(db.labels_, -1)
    assert_equal(db.core_sample_indices_.shape, (0,))

@pytest.mark.skip(reason="No functionality for use function as metric")
def test_dbscan_callable():
    # Tests the DBSCAN algorithm with a callable metric.
    # Parameters chosen specifically for this task.
    # Different eps to other test, because distance is not normalised.
    eps = 0.8
    min_samples = 10
    # metric is the function reference, not the string key.
    metric = distance.euclidean
    # Compute DBSCAN
    # parameters chosen for task
    _, labels = dbscan_sklearn(X, metric=metric, eps=eps,
                                  min_samples=min_samples,
                                  algorithm='ball_tree')

    # number of clusters, ignoring noise if present
    n_clusters_1 = len(set(labels)) - int(-1 in labels)
    assert_equal(n_clusters_1, n_clusters)

    db = DBSCAN_DAAL(metric=metric, eps=eps, min_samples=min_samples,
                algorithm='ball_tree')
    labels = db.fit(X).labels_

    n_clusters_2 = len(set(labels)) - int(-1 in labels)
    assert_equal(n_clusters_2, n_clusters)

@pytest.mark.skip(reason="Not supported Minkowski and Manhattan metrics")
def test_dbscan_metric_params():
    # Tests that DBSCAN works with the metrics_params argument.
    eps = 0.8
    min_samples = 10
    p = 1

    # Compute DBSCAN with metric_params arg
    db = DBSCAN_DAAL(metric='minkowski', metric_params={'p': p}, eps=eps,
                min_samples=min_samples, algorithm='ball_tree').fit(X)
    core_sample_1, labels_1 = db.core_sample_indices_, db.labels_

    # Test that sample labels are the same as passing Minkowski 'p' directly
    db = DBSCAN_DAAL(metric='minkowski', eps=eps, min_samples=min_samples,
                algorithm='ball_tree', p=p).fit(X)
    core_sample_2, labels_2 = db.core_sample_indices_, db.labels_

    assert_array_equal(core_sample_1, core_sample_2)
    assert_array_equal(labels_1, labels_2)

    # Minkowski with p=1 should be equivalent to Manhattan distance
    db = DBSCAN_DAAL(metric='manhattan', eps=eps, min_samples=min_samples,
                algorithm='ball_tree').fit(X)
    core_sample_3, labels_3 = db.core_sample_indices_, db.labels_

    assert_array_equal(core_sample_1, core_sample_3)
    assert_array_equal(labels_1, labels_3)

@pytest.mark.skip(reason="Not supported ball_tree and kd_tree algorithms. Not supported precomputed data processing.")
def test_dbscan_balltree():
    # Tests the DBSCAN algorithm with balltree for neighbor calculation.
    eps = 0.8
    min_samples = 10

    D = pairwise_distances(X)
    core_samples, labels = dbscan_daal(D, metric="precomputed", eps=eps,
                                  min_samples=min_samples)

    # number of clusters, ignoring noise if present
    n_clusters_1 = len(set(labels)) - int(-1 in labels)
    assert_equal(n_clusters_1, n_clusters)

    db = DBSCAN_DAAL(p=2.0, eps=eps, min_samples=min_samples, algorithm='ball_tree')
    labels = db.fit(X).labels_

    n_clusters_2 = len(set(labels)) - int(-1 in labels)
    assert_equal(n_clusters_2, n_clusters)

    db = DBSCAN_DAAL(p=2.0, eps=eps, min_samples=min_samples, algorithm='kd_tree')
    labels = db.fit(X).labels_

    n_clusters_3 = len(set(labels)) - int(-1 in labels)
    assert_equal(n_clusters_3, n_clusters)

    db = DBSCAN_DAAL(p=1.0, eps=eps, min_samples=min_samples, algorithm='ball_tree')
    labels = db.fit(X).labels_

    n_clusters_4 = len(set(labels)) - int(-1 in labels)
    assert_equal(n_clusters_4, n_clusters)

    db = DBSCAN_DAAL(leaf_size=20, eps=eps, min_samples=min_samples,
                algorithm='ball_tree')
    labels = db.fit(X).labels_

    n_clusters_5 = len(set(labels)) - int(-1 in labels)
    assert_equal(n_clusters_5, n_clusters)

@pytest.mark.skip(reason="No functionality for list processing")
def test_input_validation():
    # DBSCAN.fit should accept a list of lists.
    X = [[1., 2.], [3., 4.]]
    DBSCAN_DAAL().fit(X)             # must not raise exception

def test_pickle():
    obj = DBSCAN_DAAL()
    s = pickle.dumps(obj)
    assert_equal(type(pickle.loads(s)), obj.__class__)

def test_boundaries():
    # ensure min_samples is inclusive of core point
    core, _ = dbscan_daal([[0], [1]], eps=2, min_samples=2)
    assert_in(0, core)
    # ensure eps is inclusive of circumference
    core, _ = dbscan_daal([[0], [1], [1]], eps=1, min_samples=2)
    assert_in(0, core)
    core, _ = dbscan_daal([[0], [1], [1]], eps=.99, min_samples=2)
    assert_not_in(0, core)

def test_weighted_dbscan():
    # ensure sample_weight has an effect
    assert_array_equal([], dbscan_daal([[0], [1]], sample_weight=None,
                                  min_samples=6)[0])
    assert_array_equal([], dbscan_daal([[0], [1]], sample_weight=[5, 5],
                                  min_samples=6)[0])
    assert_array_equal([0], dbscan_daal([[0], [1]], sample_weight=[6, 5],
                                   min_samples=6)[0])
    assert_array_equal([0, 1], dbscan_daal([[0], [1]], sample_weight=[6, 6],
                                      min_samples=6)[0])

    # points within eps of each other:
    assert_array_equal([0, 1], dbscan_daal([[0], [1]], eps=1.5,
                                      sample_weight=[5, 1], min_samples=6)[0])
    # and effect of non-positive and non-integer sample_weight:
    assert_array_equal([], dbscan_daal([[0], [1]], sample_weight=[5, 0],
                                  eps=1.5, min_samples=6)[0])
    assert_array_equal([0, 1], dbscan_daal([[0], [1]], sample_weight=[5.9, 0.1],
                                      eps=1.5, min_samples=6)[0])
    assert_array_equal([0, 1], dbscan_daal([[0], [1]], sample_weight=[6, 0],
                                      eps=1.5, min_samples=6)[0])
    assert_array_equal([], dbscan_daal([[0], [1]], sample_weight=[6, -1],
                                  eps=1.5, min_samples=6)[0])

    # for non-negative sample_weight, cores should be identical to repetition
    rng = np.random.RandomState(42)
    sample_weight = rng.randint(0, 5, X.shape[0])
    core1, label1 = dbscan_daal(X, sample_weight=sample_weight)
    assert_equal(len(label1), len(X))

    X_repeated = np.repeat(X, sample_weight, axis=0)
    core_repeated, _ = dbscan_daal(X_repeated)
    core_repeated_mask = np.zeros(X_repeated.shape[0], dtype=bool)
    core_repeated_mask[core_repeated] = True
    core_mask = np.zeros(X.shape[0], dtype=bool)
    core_mask[core1] = True
    assert_array_equal(np.repeat(core_mask, sample_weight), core_repeated_mask)

    # sample_weight should work with estimator
    est = DBSCAN_DAAL().fit(X, sample_weight=sample_weight)
    core4 = est.core_sample_indices_
    label4 = est.labels_
    assert_array_equal(core1, core4)
    assert_array_equal(label1, label4)

def test_dbscan_core_samples_toy(algorithm):
    X = [[0], [2], [3], [4], [6], [8], [10]]
    n_samples = len(X)
    # Degenerate case: every sample is a core sample, either with its own
    # cluster or including other close core samples.
    core_samples, labels = dbscan_daal(X, algorithm=algorithm, eps=1,
                                  min_samples=1)
    assert_array_equal(core_samples, np.arange(n_samples))
    assert_array_equal(labels, [0, 1, 1, 1, 2, 3, 4])

    # With eps=1 and min_samples=2 only the 3 samples from the denser area
    # are core samples. All other points are isolated and considered noise.
    core_samples, labels = dbscan_daal(X, algorithm=algorithm, eps=1,
                                  min_samples=2)
    assert_array_equal(core_samples, [1, 2, 3])
    assert_array_equal(labels, [-1, 0, 0, 0, -1, -1, -1])

    # Only the sample in the middle of the dense area is core. Its two
    # neighbors are edge samples. Remaining samples are noise.
    core_samples, labels = dbscan_daal(X, algorithm=algorithm, eps=1,
                                  min_samples=3)
    assert_array_equal(core_samples, [2])
    assert_array_equal(labels, [-1, 0, 0, 0, -1, -1, -1])

    # It's no longer possible to extract core samples with eps=1:
    # everything is noise.
    core_samples, labels = dbscan_daal(X, algorithm=algorithm, eps=1,
                                  min_samples=4)
    assert_array_equal(core_samples, [])
    assert_array_equal(labels, np.full(n_samples, -1.))

@pytest.mark.skip(reason="No functionality for precomputed data processing")
def test_dbscan_precomputed_metric_with_degenerate_input_arrays():
    # see https://github.com/scikit-learn/scikit-learn/issues/4641 for
    # more details
    X = np.eye(10)
    labels = DBSCAN_DAAL(eps=0.5, metric='precomputed').fit(X).labels_
    assert_equal(len(set(labels)), 1)

    X = np.zeros((10, 10))
    labels = DBSCAN_DAAL(eps=0.5, metric='precomputed').fit(X).labels_
    assert_equal(len(set(labels)), 1)

@pytest.mark.skip(reason="No functionality for precomputed data processing")
def test_dbscan_precomputed_metric_with_initial_rows_zero():
    # sample matrix with initial two row all zero
    ar = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0],
        [0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.3],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
        [0.0, 0.0, 0.0, 0.0, 0.3, 0.1, 0.0]
    ])
    matrix = sparse.csr_matrix(ar)
    labels = DBSCAN_DAAL(eps=0.2, metric='precomputed',
                    min_samples=2).fit(matrix).labels_
    assert_array_equal(labels, [-1, -1,  0,  0,  0,  1,  1])

def check_labels_equals(left_labels: np.ndarray, right_labels: np.ndarray) -> bool:
    if left_labels.shape != right_labels.shape:
        raise Exception("Shapes not equals")
    if len(left_labels.shape) != 1:
        raise Exception("Shapes size not equals 1")
    if len(set(left_labels)) != len(set(right_labels)):
        raise Exception("Clusters count not equals")
    dict_checker = {}
    for index_sample in range(left_labels.shape[0]):
        if left_labels[index_sample] not in dict_checker:
            dict_checker[left_labels[index_sample]] = right_labels[index_sample]
        elif dict_checker[left_labels[index_sample]] != right_labels[index_sample]:
            raise Exception("Wrong clustering")
    return True

def _test_dbscan_big_data_numpy_gen(eps: float, min_samples: int, metric: str, use_weights: bool, 
                                    low=-100.0, high=100.0, samples_number=1000, sample_dimension=4):
    data, weights = generate_data(low=low, high=high, samples_number=samples_number, sample_dimension=sample_dimension)
    if use_weights is False:
        weights = None
    initialized_daal_dbscan = DBSCAN_DAAL(eps=eps, min_samples=min_samples, metric=metric).fit(X=data, sample_weight=weights)
    initialized_sklearn_dbscan = DBSCAN_SKLEARN(metric=metric, eps=eps, min_samples=min_samples).fit(X=data, sample_weight=weights)
    check_labels_equals(initialized_daal_dbscan.labels_, initialized_sklearn_dbscan.labels_)

def test_dbscan_big_data_numpy_gen(metric, use_weights):
    eps = 35.0
    min_samples = 6
    _test_dbscan_big_data_numpy_gen(eps=eps, min_samples=min_samples, metric=metric, use_weights=use_weights)

def _test_across_grid_parameter_numpy_gen(metric, use_weights: bool):
    eps_begin = 0.05
    eps_end = 0.5
    eps_step = 0.05
    min_samples_begin = 5
    min_samples_end = 15
    min_samples_step = 1
    for eps in np.arange(eps_begin, eps_end, eps_step):
        for min_samples in range(min_samples_begin, min_samples_end, min_samples_step):
            _test_dbscan_big_data_numpy_gen(eps=eps, min_samples=min_samples, metric=metric, use_weights=use_weights)

def test_across_grid_parameter_numpy_gen(metric, use_weights):
    _test_across_grid_parameter_numpy_gen(metric=metric, use_weights=use_weights)
