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


Additionally, daal4py contains code to monkey-patch the following existing scikit-learn
algorithms::

1. sklearn.linear_model.LinearRegression
2. sklearn.linear_model.Ridge
3. sklearn.linear_model.LogisticRegression(solver='lbfgs')
4. sklearn.decomposition.PCA(solver='full')
5. sklearn.cluster.KMeans(algo='full')
6. sklearn.metric.pairwise_distance()
7. sklearn.svm.SVC
