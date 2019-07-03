from daal4py import dbscan as d4p_dbscan
import numpy as np

def create_weights_by_default(number_of_examples: int):
    return np.ones([number_of_examples, 1])

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean',
                 metric_params=None, algorithm='auto', leaf_size=30, p=None,
                 n_jobs=0):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.metric_params = metric_params
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.n_jobs = n_jobs

    def fit(self, X, y=None, sample_weight=None):
        if isinstance(X, list):
            X = np.array(X)
        elements_number = X.shape[0]
        if isinstance(sample_weight, list):
            sample_weight = np.array(sample_weight)
        if sample_weight is None:
            sample_weight = create_weights_by_default(elements_number)
        if len(sample_weight.shape) == 1:
            sample_weight = np.reshape(sample_weight, (sample_weight.shape[0], 1))
        initialized_dbscan = d4p_dbscan(epsilon=self.eps, minObservations=self.min_samples, resultsToCompute="computeCoreIndices|computeCoreObservations")
        self.dbscan_result_ = initialized_dbscan.compute(data=X, weights=sample_weight)
        self.number_clusters_ = self.dbscan_result_.nClusters[0][0]
        self.labels_ = np.reshape(self.dbscan_result_.assignments, elements_number)
        if self.dbscan_result_.coreIndices is not None:
            self.number_core_samples_ = self.dbscan_result_.coreIndices.shape[0]
            self.core_sample_indices_ = np.reshape(self.dbscan_result_.coreIndices, self.number_core_samples_)
            self.components_ = self.dbscan_result_.coreObservations
        else:
            self.number_core_samples_ = 0
            self.core_sample_indices_ = np.empty(shape=(0,))
            self.components_ = np.empty(shape=(0, X.shape[1]))
        return self

def dbscan(X, sample_weight=None, eps=0.5, min_samples=5, metric='euclidean',
            metric_params=None, algorithm='auto', leaf_size=30, p=None,
            n_jobs=0):
    initialized_dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric,
                metric_params=metric_params, algorithm=algorithm, leaf_size=leaf_size, p=p,
                n_jobs=n_jobs).fit(X=X, sample_weight=sample_weight)
    return initialized_dbscan.core_sample_indices_, initialized_dbscan.labels_
