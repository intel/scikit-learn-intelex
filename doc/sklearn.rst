.. _sklearn:

#############################
Scikit-Learn API and patching
#############################

Python interface to efficient Intel(R) oneAPI Data Analytics Library provided by daal4py allows one
to create scikit-learn compatible estimators, transformers, clusterers, etc. powered by oneDAL which
are nearly as efficient as native programs.

.. _sklearn_patches:

DAAL accelerated scikit-learn
------------------------------

daal4py can dynamically patch scikit-learn estimators to use Intel(R) oneAPI Data Analytics Library
as the underlying solver, while getting the same solution faster.

It is possible to enable those patches without editing the code of a scikit-learn application by
using the following commandline flag::

    python -m daal4py my_application.py

If you are using Scikit-Learn from Intel® Distribution for Python, then
you can enable daal4py patches through an environment variable. To do this, set ``USE_DAAL4PY_SKLEARN`` to one of the values
``True``, ``'1'``, ``'y'``, ``'yes'``, ``'Y'``, ``'YES'``, ``'Yes'``, ``'true'``, ``'True'`` or ``'TRUE'`` as shown below.

On Linux and Mac OS::

    export USE_DAAL4PY_SKLEARN=1

On Windows::

    set USE_DAAL4PY_SKLEARN=1

To disable daal4py patches, set the ``USE_DAAL4PY_SKLEARN`` environment variable to 0.

Patches can also be enabled programmatically::

    import daal4py.sklearn
    daal4py.sklearn.patch_sklearn()

It is possible to undo the patch with::

    daal4py.sklearn.unpatch_sklearn()

.. _sklearn_algorithms:

Applying the monkey patch will impact the following existing scikit-learn
algorithms:

1. `sklearn.linear_model.LinearRegression <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html>`__
2. `sklearn.linear_model.Ridge <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html>`__ (solver='auto')
3. `sklearn.linear_model.LogisticRegression <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`__ and `sklearn.linear_model.LogisticRegressionCV <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html>`__ (solver in ['lbfgs', 'newton-cg'])
4. `sklearn.decomposition.PCA <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`__ (svd_solver='full', and introduces svd_solver='daal')
5. `sklearn.cluster.KMeans <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`__ (algo='full')
6. `sklearn.metric.pairwise_distance <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html>`__, with metric='cosine' or metric='correlation'
7. `sklearn.svm.SVC <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`__
8. `sklearn.linear_model.ElasticNet <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html>`__
9. `sklearn.linear_model.Lasso <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html>`__
10. `sklearn.neighbors.KNeighborsClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html>`__
11. `sklearn.neighbors.KNeighborsRegressor <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html>`__
12. `sklearn.neighbors.NearestNeighbors <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html>`__
13. `sklearn.cluster.DBSCAN <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html>`__
14. `sklearn.ensemble.RandomForestClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`__
15. `sklearn.ensemble.RandomForestRegressor <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html>`__
16. `sklearn.model_selection.train_test_split <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html>`__
17. `sklearn.manifold.TSNE <https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html>`__

Monkey-patched scikit-learn clases and functions passes scikit-learn's own test
suite, with few exceptions, specified in `deselected_tests.yaml
<https://github.com/IntelPython/daal4py/blob/master/deselected_tests.yaml>`__.

In particular the tests execute `check_estimator
<https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html>`__
on all added and monkey-patched classes, which are discovered by means of
introspection. This assures scikit-learn API compatibility of all
`daal4py.sklearn` classes.

.. _sklearn_verbose:

scikit-learn verbose
--------------------

To find out which implementation of the algorithm is currently used,
set the environment variable.

On Linux and Mac OS::

    export IDP_SKLEARN_VERBOSE=INFO

On Windows::

    set IDP_SKLEARN_VERBOSE=INFO

During the calls that use Intel-optimized scikit-learn, you will receive additional print statements
that indicate which implementation is being called.
These print statements are only available for :ref:`scikit-learn algorithms with daal4py patches <sklearn_algorithms>`.

For example, for DBSCAN you get one of these print statements depending on which implementation is used::

    INFO: sklearn.cluster.DBSCAN.fit: uses Intel(R) oneAPI Data Analytics Library solver

::

    INFO: sklearn.cluster.DBSCAN.fit: uses original Scikit-learn solver



.. _sklearn_api:

scikit-learn API
----------------

The ``daal4py.sklearn`` package contains scikit-learn compatible API which
implement a subset of scikit-learn algorithms using Intel(R) oneAPI Data Analytics Library.

Currently, these include:

1. ``daal4py.sklearn.neighbors.KNeighborsClassifier``
2. ``daal4py.sklearn.neighbors.KNeighborsRegressor``
3. ``daal4py.sklearn.neighbors.NearestNeighbors``
4. ``daal4py.sklearn.tree.DecisionTreeClassifier``
5. ``daal4py.sklearn.ensemble.RandomForestClassifier``
6. ``daal4py.sklearn.ensemble.RandomForestRegressor``
7. ``daal4py.sklearn.ensemble.AdaBoostClassifier``
8. ``daal4py.sklearn.cluster.KMeans``
9. ``daal4py.sklearn.cluster.DBSCAN``
10. ``daal4py.sklearn.decomposition.PCA``
11. ``daal4py.sklearn.linear_model.Ridge``
12. ``daal4py.sklearn.svm.SVC``
13. ``daal4py.sklearn.linear_model.logistic_regression_path``
14. ``daal4py.sklearn.linear_model.LogisticRegression``
15. ``daal4py.sklearn.linear_model.ElasticNet``
16. ``daal4py.sklearn.linear_model.Lasso``
17. ``daal4py.sklearn.model_selection._daal_train_test_split``

These classes are always available, whether the scikit-learn itself has been
patched, or not. For example::

    import daal4py.sklearn
    daal4py.sklearn.unpatch_sklearn()
    import sklearn.datasets, sklearn.svm

    digits = sklearn.datasets.load_digits()
    X, y = digits.data, digits.target

    clf_d = daal4py.sklearn.svm.SVC(kernel='rbf', gamma='scale', C = 0.5).fit(X, y)
    clf_v = sklearn.svm.SVC(kernel='rbf', gamma='scale', C =0.5).fit(X, y)

    clf_d.score(X, y) # output: 0.9905397885364496
    clf_v.score(X, y) # output: 0.9905397885364496
