##########
Algorithms
##########

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
  <https://github.intel.com/SAT/daal4py/blob/master/examples/decision_forest_classification_batch.py>`_

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
  <https://github.intel.com/SAT/daal4py/blob/master/examples/decision_tree_classification_batch.py>`_

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
  <https://github.intel.com/SAT/daal4py/blob/master/examples/gradient_boosted_classification_batch.py>`_

.. autoclass:: daal4py.gbt_classification_training
.. autoclass:: daal4py.gbt_classification_training_result
   :members:
.. autoclass:: daal4py.gbt_classification_prediction
.. autoclass:: daal4py.classifier_prediction_result
   :members:
.. autoclass:: daal4py.gbt_classification_model
   :members:

Multinomial Naive Bayes
^^^^^^^^^^^^^^^^^^^^^^^
Detailed description of parameters and semantics are described in
`Intel DAAL Naive Bayes
<https://software.intel.com/en-us/daal-programming-guide-naive-bayes-classifier>`_

Examples:

- `Single-Process Naive Bayes <https://github.intel.com/SAT/daal4py/blob/master/examples/naive_bayes_batch.py>`_
- `Multi-Process (DSPV) Naive Bayes <https://github.intel.com/SAT/daal4py/blob/master/examples/naive_bayes_dspv.py>`_
- `Multi-Process (SPMD) Naive Bayes <https://github.intel.com/SAT/daal4py/blob/master/examples/naive_bayes_spmd.py>`_

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
  <https://github.intel.com/SAT/daal4py/blob/master/examples/svm_batch.py>`_

.. autoclass:: daal4py.svm_training
.. autoclass:: daal4py.svm_training_result
   :members:
.. autoclass:: daal4py.svm_prediction
.. autoclass:: daal4py.classifier_prediction_result
   :members:
.. autoclass:: daal4py.svm_model
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
  <https://github.intel.com/SAT/daal4py/blob/master/examples/decision_forest_regression_batch.py>`_

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
  <https://github.intel.com/SAT/daal4py/blob/master/examples/decision_tree_regression_batch.py>`_

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
  <https://github.intel.com/SAT/daal4py/blob/master/examples/gradient_boosted_regression_batch.py>`_

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

- `Single-Process Linear Regression <https://github.intel.com/SAT/daal4py/blob/master/examples/linear_regression_batch.py>`_
- `Multi-Process (DSPV) Linear Regression <https://github.intel.com/SAT/daal4py/blob/master/examples/linear_regression_dspv.py>`_
- `Multi-Process (SPMD) Linear Regression <https://github.intel.com/SAT/daal4py/blob/master/examples/linear_regression_spmd.py>`_

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

- `Single-Process Ridge Regression <https://github.intel.com/SAT/daal4py/blob/master/examples/ridge_regression_batch.py>`_
- `Multi-Process (DSPV) Ridge Regression <https://github.intel.com/SAT/daal4py/blob/master/examples/ridge_regression_dspv.py>`_
- `Multi-Process (SPMD) Ridge Regression <https://github.intel.com/SAT/daal4py/blob/master/examples/ridge_regression_spmd.py>`_

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

- `Single-Process K-Means <https://github.intel.com/SAT/daal4py/blob/master/examples/kmeans_batch.py>`_
- `Multi-Process (DSPV) K-Means <https://github.intel.com/SAT/daal4py/blob/master/examples/kmeans_dspv.py>`_
- `Multi-Process (SPMD) K-Means <https://github.intel.com/SAT/daal4py/blob/master/examples/kmeans_spmd.py>`_

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

- `Single-Process Multivariate Outlier Detection <https://github.intel.com/SAT/daal4py/blob/master/examples/multivariate_outlier_batch.py>`_

.. autoclass:: daal4py.multivariate_outlier_detection
.. autoclass:: daal4py.multivariate_outlier_detection_result
   :members:

Univariate Outlier Detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Detailed description of parameters and semantics are described in
`Intel DAAL Univariate Outlier Detection <https://software.intel.com/en-us/daal-programming-guide-univariate-outlier-detection>`_

Examples:

- `Single-Process Univariate Outlier Detection <https://github.intel.com/SAT/daal4py/blob/master/examples/univariate_outlier_batch.py>`_

.. autoclass:: daal4py.univariate_outlier_detection
.. autoclass:: daal4py.univariate_outlier_detection_result
   :members:

Principal Component Analysis (PCA)
----------------------------------
Detailed description of parameters and semantics are described in
`Intel DAAL PCA <https://software.intel.com/en-us/daal-programming-guide-principal-component-analysis>`_

Examples:

- `Single-Process PCA <https://github.intel.com/SAT/daal4py/blob/master/examples/pca_batch.py>`_
- `Multi-Process (DSPV) PCA <https://github.intel.com/SAT/daal4py/blob/master/examples/pca_dspv.py>`_
- `Multi-Process (SPMD) PCA <https://github.intel.com/SAT/daal4py/blob/master/examples/pca_spmd.py>`_

.. autoclass:: daal4py.pca
.. autoclass:: daal4py.pca_result
   :members:

Singular Value Decomposition (SVD)
----------------------------------
Detailed description of parameters and semantics are described in
`Intel DAAL SVD <https://software.intel.com/en-us/daal-programming-guide-singular-value-decomposition>`_

Examples:

- `Single-Process SVD <https://github.intel.com/SAT/daal4py/blob/master/examples/svd_batch.py>`_
- `Multi-Process (DSPV) SVD <https://github.intel.com/SAT/daal4py/blob/master/examples/svd_dspv.py>`_
- `Multi-Process (SPMD) SVD <https://github.intel.com/SAT/daal4py/blob/master/examples/svd_spmd.py>`_

.. autoclass:: daal4py.svd
.. autoclass:: daal4py.svd_result
   :members:
