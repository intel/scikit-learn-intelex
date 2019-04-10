.. _sklearn:

################
Scikit-Learn API
################

Python interface to efficient Intel® Data Analytics and Acceleration Library (DAAL)
provided by daal4py allows one to create scikit-learn compatible estimators,
transformers, clusterers, etc. powered by DAAL which are nearly as efficient as
native programs.

Submodule daal4py.sklearn provides such classes.

Currently, these include::

    1. daal4py.sklearn.neighbors.KNeighborsClassifier
    2. daal4py.sklearn.ensemble.RandomForestClassifier
    3. daal4py.sklearn.ensemble.RandomForestRegressor


Additionally, daal4py contains code to monkey-patch the following existing scikit-learn algorithms:

1. `sklearn.linear_model.LinearRegression <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html>`__
2. `sklearn.linear_model.Ridge <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html>`__ (solver='auto')
3. `sklearn.linear_model.LogisticRegression <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`__ (solver='lbfgs')
4. `sklearn.decomposition.PCA <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`__ (solver='full')
5. `sklearn.cluster.KMeans <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`__ (algo='full')
6. `sklearn.metric.pairwise_distance <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html>`__, with metric='cosine' or metric='correlation'
7. `sklearn.svm.SVC <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`__

Monkey-patched scikit-learn passes scikit-learn's own test suite, with few exceptions, specified in `deselected_tests.yaml <https://github.com/IntelPython/daal4py/blob/master/deselected_tests.yaml>`__.
       
       
