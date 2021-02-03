import numpy as np
import daal4py as d4p
import logging

from daal4py.sklearn import patch_sklearn
patch_sklearn()

from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor)
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    NearestNeighbors)
from sklearn.linear_model import (
    LogisticRegression,
    LogisticRegressionCV,
    LinearRegression,
    Ridge,
    ElasticNet,
    Lasso)
from sklearn.cluster import (KMeans, DBSCAN)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.datasets import (
    make_regression,
    load_iris,
    load_diabetes)

MODELS_INFO = [
    {
        'model': KNeighborsClassifier(algorithm='brute'),
        'methods': ['kneighbors', 'predict', 'predict_proba', 'score'],
        'dataset': 'classifier',
    },
    {
        'model': KNeighborsRegressor(algorithm='brute'),
        'methods': ['kneighbors', 'predict', 'score'],
        'dataset': 'regression',
    },
    {
        'model': NearestNeighbors(algorithm='brute'),
        'methods': ['kneighbors'],
        'dataset': 'blobs',
    },
    {
        'model': DBSCAN(),
        'methods': ['fit_predict'],
        'dataset': 'blobs',
    },
    {
        'model': SVC(probability=True),
        'methods': ['decision_function', 'predict', 'predict_proba', 'score'],
        'dataset': 'classifier',
    },
    {
        'model': TSNE(),
        'methods': ['fit_transform'],
        'dataset': 'classifier',
    },
    {
        'model': KMeans(),
        'methods': ['fit_predict', 'fit_transform', 'transform', 'predict', 'score'],
        'dataset': 'blobs',
    },
    {
        'model': ElasticNet(),
        'methods': ['predict', 'score'],
        'dataset': 'regression',
    },
    {
        'model': Lasso(),
        'methods': ['predict', 'score'],
        'dataset': 'regression',
    },
    {
        'model': PCA(),
        'methods': ['fit_transform', 'transform', 'score'],
        'dataset': 'classifier',
    },
    {
        'model': RandomForestClassifier(),
        'methods': ['predict', 'predict_proba', 'predict_log_proba', 'score'],
        'dataset': 'classifier',
    },
    {
        'model': LogisticRegression(max_iter=1000, multi_class='multinomial'),
        'methods': ['decision_function', 'predict', 'predict_proba',
                    'predict_log_proba', 'score'],
        'dataset': 'classifier',
    },
    {
        'model': LogisticRegressionCV(max_iter=1000),
        'methods': ['decision_function', 'predict', 'predict_proba',
                    'predict_log_proba', 'score'],
        'dataset': 'classifier',
    },
    {
        'model': RandomForestRegressor(),
        'methods': ['predict', 'score'],
        'dataset': 'regression',
    },
    {
        'model': LinearRegression(),
        'methods': ['predict', 'score'],
        'dataset': 'regression',
    },
    {
        'model': Ridge(),
        'methods': ['predict', 'score'],
        'dataset': 'regression',
    },
]


TYPES = [
    (np.int8, 'int8'),
    (np.int16, 'int16'),
    (np.int32, 'int32'),
    (np.int64, 'int64'),
    (np.float16, 'float16'),
    (np.float32, 'float32'),
    (np.float64, 'float64'),
    (np.uint8, 'uint8'),
    (np.uint16, 'uint16'),
    (np.uint32, 'uint32'),
    (np.uint64, 'uint64'),
]


def get_class_name(x):
    return x.__class__.__name__


def generate_dataset(name, dtype, model_name):
    if model_name == 'LinearRegression':
        X, y = make_regression(n_samples=1000, n_features=5)
    elif name in ['blobs', 'classifier']:
        X, y = load_iris(return_X_y=True)
    elif name == 'regression':
        X, y = load_diabetes(return_X_y=True)
    else:
        raise ValueError('Unknown dataset type')
    X = np.array(X, dtype=dtype)
    y = np.array(y, dtype=dtype)
    return (X, y)


def run_patch(model_info, dtype):
    print(get_class_name(model_info['model']), dtype[1])
    X, y = generate_dataset(model_info['dataset'],
                            dtype[0],
                            get_class_name(model_info['model']))
    model = model_info['model']
    model.fit(X, y)
    logging.info('fit')
    for i in model_info['methods']:
        if i == 'predict':
            model.predict(X)
        elif i == 'predict_proba':
            model.predict_proba(X)
        elif i == 'predict_log_proba':
            model.predict_log_proba(X)
        elif i == 'decision_function':
            model.decision_function(X)
        elif i == 'fit_predict':
            model.fit_predict(X)
        elif i == 'transform':
            model.transform(X)
        elif i == 'fit_transform':
            model.fit_transform(X)
        elif i == 'kneighbors':
            model.kneighbors(X)
        elif i == 'score':
            model.score(X, y)
        else:
            raise ValueError(i + ' is wrong method')
        logging.info(i)


if __name__ == '__main__':
    for info in MODELS_INFO:
        for t in TYPES:
            run_patch(info, t)
