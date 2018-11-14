##########
Algorithms
##########

Principal Component Analysis (PCA)
----------------------------------
Detailed description of parameters and semantics are described in
`Intel DAAL PCA <https://software.intel.com/en-us/daal-programming-guide-principal-component-analysis>`_

Examples:

- `Single-Process PCA <https://github.com/IntelPython/daal4py/blob/master/examples/pca_batch.py>`_
- `Multi-Process PCA <https://github.com/IntelPython/daal4py/blob/master/examples/pca_spmd.py>`_

.. autoclass:: daal4py.pca
.. autoclass:: daal4py.pca_result
   :members:

Principal Component Analysis (PCA) Transform
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Detailed description of parameters and semantics are described in
`Intel DAAL PCA Transform <https://software.intel.com/en-us/daal-programming-guide-principal-components-analysis-transform>`_

Examples:

- `Single-Process PCA Transform <https://github.com/IntelPython/daal4py/blob/master/examples/pca_transform_batch.py>`_

.. autoclass:: daal4py.pca_transform
.. autoclass:: daal4py.pca_transform_result
   :members:

Singular Value Decomposition (SVD)
----------------------------------
Detailed description of parameters and semantics are described in
`Intel DAAL SVD <https://software.intel.com/en-us/daal-programming-guide-singular-value-decomposition>`_

Examples:

- `Single-Process SVD <https://github.com/IntelPython/daal4py/blob/master/examples/svd_batch.py>`_
- `Multi-Process SVD <https://github.com/IntelPython/daal4py/blob/master/examples/svd_spmd.py>`_

.. autoclass:: daal4py.svd
.. autoclass:: daal4py.svd_result
   :members:

Moments of Low Order
--------------------
Detailed description of parameters and semantics are described in
`Intel DAAL Moments of Low Order <https://software.intel.com/en-us/daal-programming-guide-moments-of-low-order>`_

Examples:

- `Single-Process Low Order Moments <https://github.com/IntelPython/daal4py/blob/master/examples/low_order_moms_dense_batch.py>`_

.. autoclass:: daal4py.low_order_moments
.. autoclass:: daal4py.low_order_moments_result
   :members:

Classification
--------------
See also `Intel DAAL Classification
<https://software.intel.com/en-us/daal-programming-guide-classification>`_.

Decision Forest Classification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Detailed description of parameters and semantics are described in
`Intel DAAL Classification Decision Forest <https://software.intel.com/en-us/daal-programming-guide-decision-forest-2>`_

Examples:

- `Single-Process Decision Forest Classification
  <https://github.com/IntelPython/daal4py/blob/master/examples/decision_forest_classification_batch.py>`_

.. autoclass:: daal4py.decision_forest_classification_training
.. autoclass:: daal4py.decision_forest_classification_training_result
   :members:
.. autoclass:: daal4py.decision_forest_classification_prediction
.. autoclass:: daal4py.classifier_prediction_result
   :members:
.. autoclass:: daal4py.decision_forest_classification_model
   :members:

Decision Tree Classification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Detailed description of parameters and semantics are described in
`Intel DAAL Classification Decision Tree <https://software.intel.com/en-us/daal-programming-guide-decision-tree-2>`_

Examples:

- `Single-Process Decision Tree Classification
  <https://github.com/IntelPython/daal4py/blob/master/examples/decision_tree_classification_batch.py>`_

.. autoclass:: daal4py.decision_tree_classification_training
.. autoclass:: daal4py.decision_tree_classification_training_result
   :members:
.. autoclass:: daal4py.decision_tree_classification_prediction
.. autoclass:: daal4py.classifier_prediction_result
   :members:
.. autoclass:: daal4py.decision_tree_classification_model
   :members:

Gradient Boosted Classification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Detailed description of parameters and semantics are described in
`Intel DAAL Classification Gradient Boosted Tree <https://software.intel.com/en-us/daal-programming-guide-gradient-boosted-tree-2>`_

Examples:

- `Single-Process Gradient Boosted Classification
  <https://github.com/IntelPython/daal4py/blob/master/examples/gradient_boosted_classification_batch.py>`_

.. autoclass:: daal4py.gbt_classification_training
.. autoclass:: daal4py.gbt_classification_training_result
   :members:
.. autoclass:: daal4py.gbt_classification_prediction
.. autoclass:: daal4py.classifier_prediction_result
   :members:
.. autoclass:: daal4py.gbt_classification_model
   :members:

k-Nearest Neighbors (kNN)
^^^^^^^^^^^^^^^^^^^^^^^^^
Detailed description of parameters and semantics are described in
`Intel DAAL k-Nearest Neighbors (kNN)
<https://software.intel.com/en-us/daal-programming-guide-k-nearest-neighbors-knn-classifier>`_

Examples:

- `Single-Process kNN
  <https://github.com/IntelPython/daal4py/blob/master/examples/kdtree_knn_classification_batch.py>`_

.. autoclass:: daal4py.kdtree_knn_classification_training
.. autoclass:: daal4py.kdtree_knn_classification_training_result
   :members:
.. autoclass:: daal4py.kdtree_knn_classification_prediction
.. autoclass:: daal4py.classifier_prediction_result
   :members:
.. autoclass:: daal4py.kdtree_knn_classification_model
   :members:

Multinomial Naive Bayes
^^^^^^^^^^^^^^^^^^^^^^^
Detailed description of parameters and semantics are described in
`Intel DAAL Naive Bayes
<https://software.intel.com/en-us/daal-programming-guide-naive-bayes-classifier>`_

Examples:

- `Single-Process Naive Bayes <https://github.com/IntelPython/daal4py/blob/master/examples/naive_bayes_batch.py>`_
- `Multi-Process Naive Bayes <https://github.com/IntelPython/daal4py/blob/master/examples/naive_bayes_spmd.py>`_

.. autoclass:: daal4py.multinomial_naive_bayes_training
.. autoclass:: daal4py.multinomial_naive_bayes_training_result
   :members:
.. autoclass:: daal4py.multinomial_naive_bayes_prediction
.. autoclass:: daal4py.classifier_prediction_result
   :members:
.. autoclass:: daal4py.multinomial_naive_bayes_model
   :members:

Support Vector Machine (SVM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Detailed description of parameters and semantics are described in
`Intel DAAL SVM <https://software.intel.com/en-us/daal-programming-guide-support-vector-machine-classifier>`_

Examples:

- `Single-Process SVM
  <https://github.com/IntelPython/daal4py/blob/master/examples/svm_batch.py>`_

.. autoclass:: daal4py.svm_training
.. autoclass:: daal4py.svm_training_result
   :members:
.. autoclass:: daal4py.svm_prediction
.. autoclass:: daal4py.classifier_prediction_result
   :members:
.. autoclass:: daal4py.svm_model
   :members:

Logistic Regression
^^^^^^^^^^^^^^^^^^^
Detailed description of parameters and semantics are described in
`Intel DAAL Logistc Regression <https://software.intel.com/en-us/daal-programming-guide-logistic-regression>`_

Examples:

- `Single-Process Binary Class Logistic Regression
  <https://github.com/IntelPython/daal4py/blob/master/examples/log_reg_binary_dense_batch.py>`_
- `Single-Process Logistic Regression
  <https://github.com/IntelPython/daal4py/blob/master/examples/log_reg_dense_batch.py>`_

.. autoclass:: daal4py.logistic_regression_training
.. autoclass:: daal4py.logistic_regression_training_result
   :members:
.. autoclass:: daal4py.logistic_regression_prediction
.. autoclass:: daal4py.classifier_prediction_result
   :members:
.. autoclass:: daal4py.logistic_regression_model
   :members:

Regression
----------
See also `Intel DAAL Regression
<https://software.intel.com/en-us/daal-programming-guide-regression>`_.

Decision Forest Regression
^^^^^^^^^^^^^^^^^^^^^^^^^^
Detailed description of parameters and semantics are described in
`Intel DAAL Regression Decision Forest <https://software.intel.com/en-us/daal-programming-guide-decision-forest-1>`_

Examples:

- `Single-Process Decision Forest Regression
  <https://github.com/IntelPython/daal4py/blob/master/examples/decision_forest_regression_batch.py>`_

.. autoclass:: daal4py.decision_forest_regression_training
.. autoclass:: daal4py.decision_forest_regression_training_result
   :members:
.. autoclass:: daal4py.decision_forest_regression_prediction
.. autoclass:: daal4py.decision_forest_regression_prediction_result
   :members:
.. autoclass:: daal4py.decision_forest_regression_model
   :members:

Decision Tree Regression
^^^^^^^^^^^^^^^^^^^^^^^^
Detailed description of parameters and semantics are described in
`Intel DAAL Regression Decision Tree <https://software.intel.com/en-us/daal-programming-guide-decision-tree-1>`_

Examples:

- `Single-Process Decision Tree Regression
  <https://github.com/IntelPython/daal4py/blob/master/examples/decision_tree_regression_batch.py>`_

.. autoclass:: daal4py.decision_tree_regression_training
.. autoclass:: daal4py.decision_tree_regression_training_result
   :members:
.. autoclass:: daal4py.decision_tree_regression_prediction
.. autoclass:: daal4py.decision_tree_regression_prediction_result
   :members:
.. autoclass:: daal4py.decision_tree_regression_model
   :members:

Gradient Boosted Regression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Detailed description of parameters and semantics are described in
`Intel DAAL Regression Gradient Boosted Tree <https://software.intel.com/en-us/daal-programming-guide-gradient-boosted-tree-1>`_

Examples:

- `Single-Process Boosted Regression Regression
  <https://github.com/IntelPython/daal4py/blob/master/examples/gradient_boosted_regression_batch.py>`_

.. autoclass:: daal4py.gbt_regression_training
.. autoclass:: daal4py.gbt_regression_training_result
   :members:
.. autoclass:: daal4py.gbt_regression_prediction
.. autoclass:: daal4py.gbt_regression_prediction_result
   :members:
.. autoclass:: daal4py.gbt_regression_model
   :members:

Linear Regression
^^^^^^^^^^^^^^^^^
Detailed description of parameters and semantics are described in
`Intel DAAL Linear and Ridge Regression <https://software.intel.com/en-us/daal-programming-guide-linear-and-ridge-regressions-computation>`_

Examples:

- `Single-Process Linear Regression <https://github.com/IntelPython/daal4py/blob/master/examples/linear_regression_batch.py>`_
- `Multi-Process Linear Regression <https://github.com/IntelPython/daal4py/blob/master/examples/linear_regression_spmd.py>`_

.. autoclass:: daal4py.linear_regression_training
.. autoclass:: daal4py.linear_regression_training_result
   :members:
.. autoclass:: daal4py.linear_regression_prediction
.. autoclass:: daal4py.linear_regression_prediction_result
   :members:
.. autoclass:: daal4py.linear_regression_model
   :members:

Ridge Regression
^^^^^^^^^^^^^^^^
Detailed description of parameters and semantics are described in
`Intel DAAL Linear and Ridge Regression <https://software.intel.com/en-us/daal-programming-guide-linear-and-ridge-regressions-computation>`_

Examples:

- `Single-Process Ridge Regression <https://github.com/IntelPython/daal4py/blob/master/examples/ridge_regression_batch.py>`_
- `Multi-Process Ridge Regression <https://github.com/IntelPython/daal4py/blob/master/examples/ridge_regression_spmd.py>`_

.. autoclass:: daal4py.ridge_regression_training
.. autoclass:: daal4py.ridge_regression_training_result
   :members:
.. autoclass:: daal4py.ridge_regression_prediction
.. autoclass:: daal4py.ridge_regression_prediction_result
   :members:
.. autoclass:: daal4py.ridge_regression_model
   :members:

K-Means Clustering
------------------
Detailed description of parameters and semantics are described in
`Intel DAAL K-Means-Clustering <https://software.intel.com/en-us/daal-programming-guide-k-means-clustering>`_

Examples:

- `Single-Process K-Means <https://github.com/IntelPython/daal4py/blob/master/examples/kmeans_batch.py>`_
- `Multi-Process K-Means <https://github.com/IntelPython/daal4py/blob/master/examples/kmeans_spmd.py>`_

K-Means Initialization
^^^^^^^^^^^^^^^^^^^^^^
Detailed description of parameters and semantics are described in
`Intel DAAL K-Means Initialization <https://software.intel.com/en-us/daal-programming-guide-initialization>`_

.. autoclass:: daal4py.kmeans_init
.. autoclass:: daal4py.kmeans_init_result
   :members:

K-Means
^^^^^^^
Detailed description of parameters and semantics are described in
`Intel DAAL K-Means Computation <https://software.intel.com/en-us/daal-programming-guide-computation>`_

.. autoclass:: daal4py.kmeans
.. autoclass:: daal4py.kmeans_result
   :members:

Outlier Detection
-----------------
Multivariate Outlier Detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Detailed description of parameters and semantics are described in
`Intel DAAL Multivariate Outlier Detection <https://software.intel.com/en-us/daal-programming-guide-multivariate-outlier-detection>`_

Examples:

- `Single-Process Multivariate Outlier Detection <https://github.com/IntelPython/daal4py/blob/master/examples/multivariate_outlier_batch.py>`_

.. autoclass:: daal4py.multivariate_outlier_detection
.. autoclass:: daal4py.multivariate_outlier_detection_result
   :members:

Univariate Outlier Detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Detailed description of parameters and semantics are described in
`Intel DAAL Univariate Outlier Detection <https://software.intel.com/en-us/daal-programming-guide-univariate-outlier-detection>`_

Examples:

- `Single-Process Univariate Outlier Detection <https://github.com/IntelPython/daal4py/blob/master/examples/univariate_outlier_batch.py>`_

.. autoclass:: daal4py.univariate_outlier_detection
.. autoclass:: daal4py.univariate_outlier_detection_result
   :members:

Optimization Solvers
--------------------
Objective Functions
^^^^^^^^^^^^^^^^^^^
Mean Squared Error Algorithm (MSE)
""""""""""""""""""""""""""""""""""
Detailed description of parameters and semantics are described in
`Intel DAAL MSE <https://software.intel.com/en-us/daal-programming-guide-mean-squared-error-algorithm>`_

Examples:
- `In Adagrad <https://github.com/IntelPython/daal4py/blob/master/examples/adagrad_mse_batch.py>`_
- `In LBFGS <https://github.com/IntelPython/daal4py/blob/master/examples/lbfgs_mse_batch.py>`_
- `In SGD <https://github.com/IntelPython/daal4py/blob/master/examples/sgd_mse_batch.py>`_

.. autoclass:: daal4py.optimization_solver_mse
.. autoclass:: daal4py.optimization_solver_mse_result
   :members:

Logistic Loss
"""""""""""""
Detailed description of parameters and semantics are described in
`Intel DAAL Logistic Loss <https://software.intel.com/en-us/daal-programming-guide-logistic-loss>`_

Examples:
- `In SGD <https://github.com/IntelPython/daal4py/blob/master/examples/sgd_logistic_loss_batch.py>`_

.. autoclass:: daal4py.optimization_solver_logistic_loss
.. autoclass:: daal4py.optimization_solver_logistic_loss_result
   :members:

Cross-entropy Loss
""""""""""""""""""
Detailed description of parameters and semantics are described in
`Intel DAAL Cross Entropy Loss <https://software.intel.com/en-us/daal-programming-guide-cross-entropy-loss>`_

.. autoclass:: daal4py.optimization_solver_cross_entropy_loss
.. autoclass:: daal4py.optimization_solver_cross_entropy_loss_result
   :members:

Iterative Solvers
^^^^^^^^^^^^^^^^^
Stochastic Gradient Descent Algorithm
"""""""""""""""""""""""""""""""""""""
Detailed description of parameters and semantics are described in
`Intel DAAL SGD <https://software.intel.com/en-us/daal-programming-guide-stochastic-gradient-descent-algorithm>`_

Examples:
- `Using Logistic Loss <https://github.com/IntelPython/daal4py/blob/master/examples/sgd_logistic_loss_batch.py>`_
- `Using MSE <https://github.com/IntelPython/daal4py//blob/master/examples/sgd_mse_batch.py>`_

.. autoclass:: daal4py.optimization_solver_sgd
.. autoclass:: daal4py.optimization_solver_sgd_result
   :members:

Limited-Memory Broyden-Fletcher-Goldfarb-Shanno Algorithm
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Detailed description of parameters and semantics are described in
`Intel DAAL LBFGS <https://software.intel.com/en-us/daal-programming-guide-limited-memory-broyden-fletcher-goldfarb-shanno-algorithm>`_

Examples:
- `Using MSE <https://github.com/IntelPython/daal4py/blob/master/examples/lbfgs_mse_batch.py>`_

.. autoclass:: daal4py.optimization_solver_lbfgs
.. autoclass:: daal4py.optimization_solver_lbfgs_result
   :members:

Adaptive Subgradient Method
"""""""""""""""""""""""""""
Detailed description of parameters and semantics are described in
`Intel DAAL AdaGrad <https://software.intel.com/en-us/daal-programming-guide-adaptive-subgradient-method>`_

Examples:
- `Using MSE <https://github.com/IntelPython/daal4py/blob/master/examples/adagrad_mse_batch.py>`_

.. autoclass:: daal4py.optimization_solver_adagrad
.. autoclass:: daal4py.optimization_solver_adagrad_result
   :members:

Distances
---------
Cosine Distance Matrix
^^^^^^^^^^^^^^^^^^^^^^
Detailed description of parameters and semantics are described in
`Intel DAAL Cosine Distance <https://software.intel.com/en-us/daal-programming-guide-cosine-distance-matrix>`_

Examples:

- `Single-Process Cosine Distance <https://github.com/IntelPython/daal4py/blob/master/examples/cosine_distance_batch.py>`_

.. autoclass:: daal4py.cosine_distance
.. autoclass:: daal4py.cosine_distance_result
   :members:

Correlation Distance Matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Detailed description of parameters and semantics are described in
`Intel DAAL Correlation Distance <https://software.intel.com/en-us/daal-programming-guide-correlation-distance-matrix>`_

Examples:

- `Single-Process Correlation Distance <https://github.com/IntelPython/daal4py/blob/master/examples/correlation_distance_batch.py>`_

.. autoclass:: daal4py.correlation_distance
.. autoclass:: daal4py.correlation_distance_result
   :members:

Trees
-----
.. autofunction:: daal4py.getTreeState

Examples:

- `Decision Forest Regression <https://github.com/IntelPython/daal4py/blob/master/examples/decision_forest_regression_traverse_batch.py>`_
- `Decision Forest Classification <https://github.com/IntelPython/daal4py/blob/master/examples/decision_forest_classification_traverse_batch.py>`_
- `Decision Tree Regression <https://github.com/IntelPython/daal4py/blob/master/examples/decision_tree_regression_traverse_batch.py>`_
- `Decision Tree Classification <https://github.com/IntelPython/daal4py/blob/master/examples/decision_tree_classification_traverse_batch.py>`_
- `Gradient Boosted Trees Regression <https://github.com/IntelPython/daal4py/blob/master/examples/gradient_boosted_regression_traverse_batch.py>`_
- `Gradient Boosted Trees Classification <https://github.com/IntelPython/daal4py/blob/master/examples/gradient_boosted_classification_traverse_batch.py>`_
