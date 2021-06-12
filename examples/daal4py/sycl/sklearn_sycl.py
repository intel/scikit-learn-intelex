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

# daal4py Scikit-Learn examples for GPU
# run like this:
#    python -m daal4py ./sklearn_sycl.py

import numpy as np

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import DBSCAN

from sklearn.datasets import load_iris

dpctx_available = False
try:
    from dpctx import device_context, device_type
    dpctx_available = True
except ImportError:
    try:
        from daal4py.oneapi import sycl_context
        sycl_extention_available = True
    except:
        sycl_extention_available = False

gpu_available = False
if dpctx_available:
    try:
        with device_context(device_type.gpu, 0):
            gpu_available = True
    except:
        gpu_available = False

elif sycl_extention_available:
    try:
        with sycl_context('gpu'):
            gpu_available = True
    except:
        gpu_available = False


def k_means_init_x():
    print("KMeans init=X[:2]")
    X = np.array([[1., 2.], [1., 4.], [1., 0.],
                  [10., 2.], [10., 4.], [10., 0.]], dtype=np.float32)
    kmeans = KMeans(n_clusters=2, random_state=0, init=X[:2]).fit(X)
    print("kmeans.labels_")
    print(kmeans.labels_)
    print("kmeans.predict([[0, 0], [12, 3]])")
    print(kmeans.predict(np.array([[0, 0], [12, 3]], dtype=np.float32)))
    print("kmeans.cluster_centers_")
    print(kmeans.cluster_centers_)


def k_means_random():
    print("KMeans init='random'")
    X = np.array([[1., 2.], [1., 4.], [1., 0.],
                  [10., 2.], [10., 4.], [10., 0.]], dtype=np.float32)
    kmeans = KMeans(n_clusters=2, random_state=0, init='random').fit(X)
    print("kmeans.labels_")
    print(kmeans.labels_)
    print("kmeans.predict([[0, 0], [12, 3]])")
    print(kmeans.predict(np.array([[0, 0], [12, 3]], dtype=np.float32)))
    print("kmeans.cluster_centers_")
    print(kmeans.cluster_centers_)


def linear_regression():
    print("LinearRegression")
    X = np.array([[1., 1.], [1., 2.], [2., 2.], [2., 3.]], dtype=np.float32)
    # y = 1 * x_0 + 2 * x_1 + 3
    y = np.dot(X, np.array([1, 2], dtype=np.float32)) + 3
    reg = LinearRegression().fit(X, y)
    print("reg.score(X, y)")
    print(reg.score(X, y))
    print("reg.coef_")
    print(reg.coef_)
    print("reg.intercept_")
    print(reg.intercept_)
    print("reg.predict(np.array([[3, 5]], dtype=np.float32))")
    print(reg.predict(np.array([[3, 5]], dtype=np.float32)))


def logistic_regression_lbfgs():
    print("LogisticRegression solver='lbfgs'")
    X, y = load_iris(return_X_y=True)
    clf = LogisticRegression(random_state=0, solver='lbfgs').fit(
        X.astype('float32'),
        y.astype('float32'))
    print("clf.predict(X[:2, :])")
    print(clf.predict(X[:2, :]))
    print("clf.predict_proba(X[:2, :])")
    print(clf.predict_proba(X[:2, :]))
    print("clf.score(X, y)")
    print(clf.score(X, y))


def logistic_regression_newton():
    print("LogisticRegression solver='newton-cg'")
    X, y = load_iris(return_X_y=True)
    clf = LogisticRegression(random_state=0, solver='newton-cg').fit(
        X.astype('float32'),
        y.astype('float32'))
    print("clf.predict(X[:2, :])")
    print(clf.predict(X[:2, :]))
    print("clf.predict_proba(X[:2, :])")
    print(clf.predict_proba(X[:2, :]))
    print("clf.score(X, y)")
    print(clf.score(X, y))


def dbscan():
    print("DBSCAN")
    X = np.array([[1., 2.], [2., 2.], [2., 3.],
                  [8., 7.], [8., 8.], [25., 80.]], dtype=np.float32)
    clustering = DBSCAN(eps=3, min_samples=2).fit(X)
    print("clustering.labels_")
    print(clustering.labels_)
    print("clustering")
    print(clustering)


def get_context(device):
    if dpctx_available:
        return device_context(device, 0)
    if sycl_extention_available:
        return sycl_context(device)
    return None


if __name__ == "__main__":
    examples = [
        k_means_init_x,
        k_means_random,
        linear_regression,
        logistic_regression_lbfgs,
        logistic_regression_newton,
        dbscan,
    ]
    devices = []

    if dpctx_available:
        devices.append(device_type.host)
        devices.append(device_type.cpu)
        if gpu_available:
            devices.append(device_type.gpu)

    elif sycl_extention_available:
        devices.append('host')
        devices.append('cpu')
        if gpu_available:
            devices.append('gpu')

    for device in devices:
        for e in examples:
            print("*" * 80)
            print("device context:", device)
            with get_context(device):
                e()
            print("*" * 80)

    print('All looks good!')
