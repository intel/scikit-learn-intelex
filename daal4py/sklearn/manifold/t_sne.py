from ..neighbors import NearestNeighbors
from sklearn.manifold import TSNE as Base_TSNE

class TSNE(Base_TSNE):
    def __init__(self, n_components=2, perplexity=30.0,
                 early_exaggeration=12.0, learning_rate=200.0, n_iter=1000,
                 n_iter_without_progress=300, min_grad_norm=1e-7,
                 metric="euclidean", init="random", verbose=0,
                 random_state=None, method='barnes_hut', angle=0.5,
                 n_jobs=None):
        print(f"TSNE DAAL constructor")
        super().__init__(
            n_components=n_components,  perplexity=perplexity,
                 early_exaggeration=early_exaggeration, learning_rate=learning_rate, n_iter=n_iter,
                 n_iter_without_progress=n_iter_without_progress, min_grad_norm=min_grad_norm,
                 metric=metric, init=init, verbose=verbose,
                 random_state=random_state, method=method, angle=angle,
                 n_jobs=n_jobs
        )