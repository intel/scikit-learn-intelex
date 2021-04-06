##########
Algorithms
##########

Classification
--------------
See also |onedal-dg-classification|_.

Decision Forest Classification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Parameters and semantics are described in |onedal-dg-classification-decision-forest|_.

.. rubric:: Examples:

- `Single-Process Decision Forest Classification Default Dense method
  <https://github.com/IntelPython/daal4py/blob/master/examples/decision_forest_classification_default_dense_batch.py>`__
- `Single-Process Decision Forest Classification Histogram method
  <https://github.com/IntelPython/daal4py/blob/master/examples/decision_forest_classification_hist_batch.py>`__

.. autoclass:: daal4py.decision_forest_classification_training
   :members: compute
.. autoclass:: daal4py.decision_forest_classification_training_result
   :members:
.. autoclass:: daal4py.decision_forest_classification_prediction
   :members: compute
.. autoclass:: daal4py.classifier_prediction_result
   :members:
.. autoclass:: daal4py.decision_forest_classification_model
   :members:

Decision Tree Classification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Parameters and semantics are described in |onedal-dg-classification-decision-tree|_.

.. rubric:: Examples:

- `Single-Process Decision Tree Classification
  <https://github.com/IntelPython/daal4py/blob/master/examples/decision_tree_classification_batch.py>`__

.. autoclass:: daal4py.decision_tree_classification_training
   :members: compute
.. autoclass:: daal4py.decision_tree_classification_training_result
   :members:
.. autoclass:: daal4py.decision_tree_classification_prediction
   :members: compute
.. autoclass:: daal4py.classifier_prediction_result
   :members:
.. autoclass:: daal4py.decision_tree_classification_model
   :members:

Gradient Boosted Classification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Parameters and semantics are described in |onedal-dg-classification-gradient-boosted-tree|_.

.. rubric:: Examples:

- `Single-Process Gradient Boosted Classification
  <https://github.com/IntelPython/daal4py/blob/master/examples/gradient_boosted_classification_batch.py>`__

.. autoclass:: daal4py.gbt_classification_training
   :members: compute
.. autoclass:: daal4py.gbt_classification_training_result
   :members:
.. autoclass:: daal4py.gbt_classification_prediction
   :members: compute
.. autoclass:: daal4py.classifier_prediction_result
   :members:
.. autoclass:: daal4py.gbt_classification_model
   :members:

k-Nearest Neighbors (kNN)
^^^^^^^^^^^^^^^^^^^^^^^^^
Parameters and semantics are described in |onedal-dg-k-nearest-neighbors-knn|_.

.. rubric:: Examples:

- `Single-Process kNN
  <https://github.com/IntelPython/daal4py/blob/master/examples/kdtree_knn_classification_batch.py>`__

.. autoclass:: daal4py.kdtree_knn_classification_training
   :members: compute
.. autoclass:: daal4py.kdtree_knn_classification_training_result
   :members:
.. autoclass:: daal4py.kdtree_knn_classification_prediction
   :members: compute
.. autoclass:: daal4py.classifier_prediction_result
   :members:
.. autoclass:: daal4py.kdtree_knn_classification_model
   :members:

Brute-force k-Nearest Neighbors (kNN)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Parameters and semantics are described in |onedal-dg-k-nearest-neighbors-knn|_.

.. autoclass:: daal4py.bf_knn_classification_training
   :members: compute
.. autoclass:: daal4py.bf_knn_classification_training_result
   :members:
.. autoclass:: daal4py.bf_knn_classification_prediction
   :members: compute
.. autoclass:: daal4py.classifier_prediction_result
   :members:
.. autoclass:: daal4py.bf_knn_classification_model
   :members:

AdaBoost Classification
^^^^^^^^^^^^^^^^^^^^^^^
Parameters and semantics are described in |onedal-dg-classification-adaboost|_.

.. rubric:: Examples:

- `Single-Process AdaBoost Classification
  <https://github.com/IntelPython/daal4py/blob/master/examples/adaboost_batch.py>`__

.. autoclass:: daal4py.adaboost_training
   :members: compute
.. autoclass:: daal4py.adaboost_training_result
   :members:
.. autoclass:: daal4py.adaboost_prediction
   :members: compute
.. autoclass:: daal4py.classifier_prediction_result
   :members:
.. autoclass:: daal4py.adaboost_model
   :members:

BrownBoost Classification
^^^^^^^^^^^^^^^^^^^^^^^^^
Parameters and semantics are described in |onedal-dg-classification-brownboost|_.

.. rubric:: Examples:

- `Single-Process BrownBoost Classification
  <https://github.com/IntelPython/daal4py/blob/master/examples/brownboost_batch.py>`__

.. autoclass:: daal4py.brownboost_training
   :members: compute
.. autoclass:: daal4py.brownboost_training_result
   :members:
.. autoclass:: daal4py.brownboost_prediction
   :members: compute
.. autoclass:: daal4py.classifier_prediction_result
   :members:
.. autoclass:: daal4py.brownboost_model
   :members:

LogitBoost Classification
^^^^^^^^^^^^^^^^^^^^^^^^^
Parameters and semantics are described in |onedal-dg-classification-logitboost|_.

.. rubric:: Examples:

- `Single-Process LogitBoost Classification
  <https://github.com/IntelPython/daal4py/blob/master/examples/logitboost_batch.py>`__

.. autoclass:: daal4py.logitboost_training
   :members: compute
.. autoclass:: daal4py.logitboost_training_result
   :members:
.. autoclass:: daal4py.logitboost_prediction
   :members: compute
.. autoclass:: daal4py.classifier_prediction_result
   :members:
.. autoclass:: daal4py.logitboost_model
   :members:

Stump Weak Learner Classification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Parameters and semantics are described in |onedal-dg-classification-weak-learner-stump|_.

.. rubric:: Examples:

- `Single-Process Stump Weak Learner Classification
  <https://github.com/IntelPython/daal4py/blob/master/examples/stump_classification_batch.py>`__

.. autoclass:: daal4py.stump_classification_training
   :members: compute
.. autoclass:: daal4py.stump_classification_training_result
   :members:
.. autoclass:: daal4py.stump_classification_prediction
   :members: compute
.. autoclass:: daal4py.classifier_prediction_result
   :members:
.. autoclass:: daal4py.stump_classification_model
   :members:

Multinomial Naive Bayes
^^^^^^^^^^^^^^^^^^^^^^^
Parameters and semantics are described in |onedal-dg-naive-bayes|_.

.. rubric:: Examples:

- `Single-Process Naive Bayes <https://github.com/IntelPython/daal4py/blob/master/examples/naive_bayes_batch.py>`__
- `Streaming Naive Bayes <https://github.com/IntelPython/daal4py/blob/master/examples/naive_bayes_streaming.py>`__
- `Multi-Process Naive Bayes <https://github.com/IntelPython/daal4py/blob/master/examples/naive_bayes_spmd.py>`__

.. autoclass:: daal4py.multinomial_naive_bayes_training
   :members: compute
.. autoclass:: daal4py.multinomial_naive_bayes_training_result
   :members:
.. autoclass:: daal4py.multinomial_naive_bayes_prediction
   :members: compute
.. autoclass:: daal4py.classifier_prediction_result
   :members:
.. autoclass:: daal4py.multinomial_naive_bayes_model
   :members:

Support Vector Machine (SVM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Parameters and semantics are described in |onedal-dg-svm|_.

Note: For the labels parameter, data is formatted as -1s and 1s

.. rubric:: Examples:

- `Single-Process SVM
  <https://github.com/IntelPython/daal4py/blob/master/examples/svm_batch.py>`__

.. autoclass:: daal4py.svm_training
   :members: compute
.. autoclass:: daal4py.svm_training_result
   :members:
.. autoclass:: daal4py.svm_prediction
   :members: compute
.. autoclass:: daal4py.classifier_prediction_result
   :members:
.. autoclass:: daal4py.svm_model
   :members:

Logistic Regression
^^^^^^^^^^^^^^^^^^^
Parameters and semantics are described in |onedal-dg-logistic-regression|_.

.. rubric:: Examples:

- `Single-Process Binary Class Logistic Regression
  <https://github.com/IntelPython/daal4py/blob/master/examples/log_reg_binary_dense_batch.py>`__
- `Single-Process Logistic Regression
  <https://github.com/IntelPython/daal4py/blob/master/examples/log_reg_dense_batch.py>`__

.. autoclass:: daal4py.logistic_regression_training
   :members: compute
.. autoclass:: daal4py.logistic_regression_training_result
   :members:
.. autoclass:: daal4py.logistic_regression_prediction
   :members: compute
.. autoclass:: daal4py.classifier_prediction_result
   :members:
.. autoclass:: daal4py.logistic_regression_model
   :members:

Regression
----------
See also |onedal-dg-regression|_.

Decision Forest Regression
^^^^^^^^^^^^^^^^^^^^^^^^^^
Parameters and semantics are described in |onedal-dg-regression-decision-forest|_.

.. rubric:: Examples:

- `Single-Process Decision Forest Regression Default Dense method
  <https://github.com/IntelPython/daal4py/blob/master/examples/decision_forest_regression_default_dense_batch.py>`__
- `Single-Process Decision Forest Regression Histogram method
  <https://github.com/IntelPython/daal4py/blob/master/examples/decision_forest_regression_hist_batch.py>`__

.. autoclass:: daal4py.decision_forest_regression_training
   :members: compute
.. autoclass:: daal4py.decision_forest_regression_training_result
   :members:
.. autoclass:: daal4py.decision_forest_regression_prediction
   :members: compute
.. autoclass:: daal4py.decision_forest_regression_prediction_result
   :members:
.. autoclass:: daal4py.decision_forest_regression_model
   :members:

Decision Tree Regression
^^^^^^^^^^^^^^^^^^^^^^^^
Parameters and semantics are described in |onedal-dg-regression-decision-tree|_.

.. rubric:: Examples:

- `Single-Process Decision Tree Regression
  <https://github.com/IntelPython/daal4py/blob/master/examples/decision_tree_regression_batch.py>`__

.. autoclass:: daal4py.decision_tree_regression_training
   :members: compute
.. autoclass:: daal4py.decision_tree_regression_training_result
   :members:
.. autoclass:: daal4py.decision_tree_regression_prediction
   :members: compute
.. autoclass:: daal4py.decision_tree_regression_prediction_result
   :members:
.. autoclass:: daal4py.decision_tree_regression_model
   :members:

Gradient Boosted Regression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Parameters and semantics are described in |onedal-dg-regression-gradient-boosted-tree|_.

.. rubric:: Examples:

- `Single-Process Boosted Regression Regression
  <https://github.com/IntelPython/daal4py/blob/master/examples/gradient_boosted_regression_batch.py>`__

.. autoclass:: daal4py.gbt_regression_training
   :members: compute
.. autoclass:: daal4py.gbt_regression_training_result
   :members:
.. autoclass:: daal4py.gbt_regression_prediction
   :members: compute
.. autoclass:: daal4py.gbt_regression_prediction_result
   :members:
.. autoclass:: daal4py.gbt_regression_model
   :members:

Linear Regression
^^^^^^^^^^^^^^^^^
Parameters and semantics are described in |onedal-dg-linear-regression|_.

.. rubric:: Examples:

- `Single-Process Linear Regression <https://github.com/IntelPython/daal4py/blob/master/examples/linear_regression_batch.py>`__
- `Streaming Linear Regression <https://github.com/IntelPython/daal4py/blob/master/examples/linear_regression_streaming.py>`__
- `Multi-Process Linear Regression <https://github.com/IntelPython/daal4py/blob/master/examples/linear_regression_spmd.py>`__

.. autoclass:: daal4py.linear_regression_training
   :members: compute
.. autoclass:: daal4py.linear_regression_training_result
   :members:
.. autoclass:: daal4py.linear_regression_prediction
   :members: compute
.. autoclass:: daal4py.linear_regression_prediction_result
   :members:
.. autoclass:: daal4py.linear_regression_model
   :members:

Least Absolute Shrinkage and Selection Operator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Parameters and semantics are described in |onedal-dg-least-absolute-shrinkage-and-selection-operator|_.

.. rubric:: Examples:

- `Single-Process LASSO Regression <https://github.com/IntelPython/daal4py/blob/master/examples/lasso_regression_batch.py>`__

.. autoclass:: daal4py.lasso_regression_training
   :members: compute
.. autoclass:: daal4py.lasso_regression_training_result
   :members:
.. autoclass:: daal4py.lasso_regression_prediction
   :members: compute
.. autoclass:: daal4py.lasso_regression_prediction_result
   :members:
.. autoclass:: daal4py.lasso_regression_model
   :members:

Ridge Regression
^^^^^^^^^^^^^^^^
Parameters and semantics are described in |onedal-dg-ridge-regression|_.

.. rubric:: Examples:

- `Single-Process Ridge Regression <https://github.com/IntelPython/daal4py/blob/master/examples/ridge_regression_batch.py>`__
- `Streaming Ridge Regression <https://github.com/IntelPython/daal4py/blob/master/examples/ridge_regression_streaming.py>`__
- `Multi-Process Ridge Regression <https://github.com/IntelPython/daal4py/blob/master/examples/ridge_regression_spmd.py>`__

.. autoclass:: daal4py.ridge_regression_training
   :members: compute
.. autoclass:: daal4py.ridge_regression_training_result
   :members:
.. autoclass:: daal4py.ridge_regression_prediction
   :members: compute
.. autoclass:: daal4py.ridge_regression_prediction_result
   :members:
.. autoclass:: daal4py.ridge_regression_model
   :members:

Stump Regression
^^^^^^^^^^^^^^^^
Parameters and semantics are described in |onedal-dg-regression-stump|_.

.. rubric:: Examples:

- `Single-Process Stump Regression
  <https://github.com/IntelPython/daal4py/blob/master/examples/stump_regression_batch.py>`__

.. autoclass:: daal4py.stump_regression_training
   :members: compute
.. autoclass:: daal4py.stump_regression_training_result
   :members:
.. autoclass:: daal4py.stump_regression_prediction
   :members: compute
.. autoclass:: daal4py.stump_regression_prediction_result
   :members:
.. autoclass:: daal4py.stump_regression_model
   :members:

Principal Component Analysis (PCA)
----------------------------------
Parameters and semantics are described in |onedal-dg-pca|_.

.. rubric:: Examples:

- `Single-Process PCA <https://github.com/IntelPython/daal4py/blob/master/examples/pca_batch.py>`__
- `Multi-Process PCA <https://github.com/IntelPython/daal4py/blob/master/examples/pca_spmd.py>`__

.. autoclass:: daal4py.pca
   :members: compute
.. autoclass:: daal4py.pca_result
   :members:

Principal Component Analysis (PCA) Transform
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Parameters and semantics are described in |onedal-dg-pca-transform|_.

.. rubric:: Examples:

- `Single-Process PCA Transform <https://github.com/IntelPython/daal4py/blob/master/examples/pca_transform_batch.py>`__

.. autoclass:: daal4py.pca_transform
   :members: compute
.. autoclass:: daal4py.pca_transform_result
   :members:

K-Means Clustering
------------------
Parameters and semantics are described in |onedal-dg-k-means-clustering|_.

.. rubric:: Examples:

- `Single-Process K-Means <https://github.com/IntelPython/daal4py/blob/master/examples/kmeans_batch.py>`__
- `Multi-Process K-Means <https://github.com/IntelPython/daal4py/blob/master/examples/kmeans_spmd.py>`__

K-Means Initialization
^^^^^^^^^^^^^^^^^^^^^^
Parameters and semantics are described in |onedal-dg-k-means-initialization|_.

.. autoclass:: daal4py.kmeans_init
   :members: compute
.. autoclass:: daal4py.kmeans_init_result
   :members:

K-Means
^^^^^^^
Parameters and semantics are described in |onedal-dg-k-means-computation|_.

.. autoclass:: daal4py.kmeans
   :members: compute
.. autoclass:: daal4py.kmeans_result
   :members:

Density-Based Spatial Clustering of Applications with Noise
-----------------------------------------------------------
Parameters and semantics are described in |onedal-dg-density-based-spatial-clustering-of-applications-with-noise|_.

.. rubric:: Examples:

- `Single-Process DBSCAN <https://github.com/IntelPython/daal4py/blob/master/examples/dbscan_batch.py>`__

.. autoclass:: daal4py.dbscan
   :members: compute
.. autoclass:: daal4py.dbscan_result
   :members:

Outlier Detection
-----------------
Multivariate Outlier Detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Parameters and semantics are described in |onedal-dg-multivariate-outlier-detection|_.

.. rubric:: Examples:

- `Single-Process Multivariate Outlier Detection <https://github.com/IntelPython/daal4py/blob/master/examples/multivariate_outlier_batch.py>`__

.. autoclass:: daal4py.multivariate_outlier_detection
   :members: compute
.. autoclass:: daal4py.multivariate_outlier_detection_result
   :members:

Univariate Outlier Detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Parameters and semantics are described in |onedal-dg-univariate-outlier-detection|_.

.. rubric:: Examples:

- `Single-Process Univariate Outlier Detection <https://github.com/IntelPython/daal4py/blob/master/examples/univariate_outlier_batch.py>`__

.. autoclass:: daal4py.univariate_outlier_detection
   :members: compute
.. autoclass:: daal4py.univariate_outlier_detection_result
   :members:

Multivariate Bacon Outlier Detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Parameters and semantics are described in |onedal-dg-multivariate-bacon-outlier-detection|_.

.. rubric:: Examples:

- `Single-Process Bacon Outlier Detection <https://github.com/IntelPython/daal4py/blob/master/examples/bacon_outlier_batch.py>`__

.. autoclass:: daal4py.bacon_outlier_detection
   :members: compute
.. autoclass:: daal4py.bacon_outlier_detection_result
   :members:

Optimization Solvers
--------------------
Objective Functions
^^^^^^^^^^^^^^^^^^^
Mean Squared Error Algorithm (MSE)
""""""""""""""""""""""""""""""""""
Parameters and semantics are described in |onedal-dg-mse|_.

.. rubric:: Examples:
- `In Adagrad <https://github.com/IntelPython/daal4py/blob/master/examples/adagrad_mse_batch.py>`__
- `In LBFGS <https://github.com/IntelPython/daal4py/blob/master/examples/lbfgs_mse_batch.py>`__
- `In SGD <https://github.com/IntelPython/daal4py/blob/master/examples/sgd_mse_batch.py>`__

.. autoclass:: daal4py.optimization_solver_mse
   :members: compute, setup
.. autoclass:: daal4py.optimization_solver_mse_result
   :members:

Logistic Loss
"""""""""""""
Parameters and semantics are described in |onedal-dg-logistic-loss|_.

.. rubric:: Examples:
- `In SGD <https://github.com/IntelPython/daal4py/blob/master/examples/sgd_logistic_loss_batch.py>`__

.. autoclass:: daal4py.optimization_solver_logistic_loss
   :members: compute, setup
.. autoclass:: daal4py.optimization_solver_logistic_loss_result
   :members:

Cross-entropy Loss
""""""""""""""""""
Parameters and semantics are described in |onedal-dg-cross-entropy-loss|_.

.. rubric:: Examples:
- `In LBFGS <https://github.com/IntelPython/daal4py/blob/master/examples/lbfgs_cr_entr_loss_batch.py>`__

.. autoclass:: daal4py.optimization_solver_cross_entropy_loss
   :members: compute, setup
.. autoclass:: daal4py.optimization_solver_cross_entropy_loss_result
   :members:

Iterative Solvers
^^^^^^^^^^^^^^^^^
Stochastic Gradient Descent Algorithm
"""""""""""""""""""""""""""""""""""""
Parameters and semantics are described in |onedal-dg-sgd|_.

.. rubric:: Examples:
- `Using Logistic Loss <https://github.com/IntelPython/daal4py/blob/master/examples/sgd_logistic_loss_batch.py>`__
- `Using MSE <https://github.com/IntelPython/daal4py//blob/master/examples/sgd_mse_batch.py>`__

.. autoclass:: daal4py.optimization_solver_sgd
   :members: compute
.. autoclass:: daal4py.optimization_solver_sgd_result
   :members:

Limited-Memory Broyden-Fletcher-Goldfarb-Shanno Algorithm
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Parameters and semantics are described in |onedal-dg-lbfgs|_.

.. rubric:: Examples:
- `Using MSE <https://github.com/IntelPython/daal4py/blob/master/examples/lbfgs_mse_batch.py>`__

.. autoclass:: daal4py.optimization_solver_lbfgs
   :members: compute
.. autoclass:: daal4py.optimization_solver_lbfgs_result
   :members:

Adaptive Subgradient Method
"""""""""""""""""""""""""""
Parameters and semantics are described in |onedal-dg-adagrad|_.

.. rubric:: Examples:
- `Using MSE <https://github.com/IntelPython/daal4py/blob/master/examples/adagrad_mse_batch.py>`__

.. autoclass:: daal4py.optimization_solver_adagrad
   :members: compute
.. autoclass:: daal4py.optimization_solver_adagrad_result
   :members:

Stochastic Average Gradient Descent
"""""""""""""""""""""""""""""""""""
Parameters and semantics are described in |onedal-dg-stochastic-average-gradient-descent-saga|_.

.. rubric:: Examples:
- `Single Proces saga-logistc_loss <https://github.com/IntelPython/daal4py/blob/master/examples/saga_batch.py>`__

.. autoclass:: daal4py.optimization_solver_saga
   :members: compute
.. autoclass:: daal4py.optimization_solver_saga_result
   :members:

Distances
---------
Cosine Distance Matrix
^^^^^^^^^^^^^^^^^^^^^^
Parameters and semantics are described in |onedal-dg-cosine-distance|_.

.. rubric:: Examples:

- `Single-Process Cosine Distance <https://github.com/IntelPython/daal4py/blob/master/examples/cosine_distance_batch.py>`__

.. autoclass:: daal4py.cosine_distance
   :members: compute
.. autoclass:: daal4py.cosine_distance_result
   :members:

Correlation Distance Matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Parameters and semantics are described in |onedal-dg-correlation-distance|_.

.. rubric:: Examples:

- `Single-Process Correlation Distance <https://github.com/IntelPython/daal4py/blob/master/examples/correlation_distance_batch.py>`__

.. autoclass:: daal4py.correlation_distance
   :members: compute
.. autoclass:: daal4py.correlation_distance_result
   :members:

Expectation-Maximization (EM)
-----------------------------
Parameters and semantics are described in |onedal-dg-expectation-maximization|_.

Initialization for the Gaussian Mixture Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Parameters and semantics are described in |onedal-dg-expectation-maximization-initialization|_.

.. rubric:: Examples:

- `Single-Process Expectation-Maximization <https://github.com/IntelPython/daal4py/blob/master/examples/em_gmm_batch.py>`__

.. autoclass:: daal4py.em_gmm_init
   :members: compute
.. autoclass:: daal4py.em_gmm_init_result
   :members:

EM algorithm for the Gaussian Mixture Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Parameters and semantics are described in |onedal-dg-expectation-maximization-for-the-gaussian-mixture-model|_.

.. rubric:: Examples:

- `Single-Process Expectation-Maximization <https://github.com/IntelPython/daal4py/blob/master/examples/em_gmm_batch.py>`__

.. autoclass:: daal4py.em_gmm
   :members: compute
.. autoclass:: daal4py.em_gmm_result
   :members:

QR Decomposition
----------------
Parameters and semantics are described in |onedal-dg-qr-decomposition|_.

QR Decomposition (without pivoting)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Parameters and semantics are described in |onedal-dg-qr-decomposition-without-pivoting|_.

.. rubric:: Examples:

- `Single-Process QR <https://github.com/IntelPython/daal4py/blob/master/examples/qr_batch.py>`__
- `Streaming QR <https://github.com/IntelPython/daal4py/blob/master/examples/qr_streaming.py>`__

.. autoclass:: daal4py.qr
   :members: compute
.. autoclass:: daal4py.qr_result
   :members:

Pivoted QR Decomposition
^^^^^^^^^^^^^^^^^^^^^^^^
Parameters and semantics are described in |onedal-dg-pivoted-qr-decomposition|_.

.. rubric:: Examples:

- `Single-Process Pivoted QR <https://github.com/IntelPython/daal4py/blob/master/examples/pivoted_qr_batch.py>`__

.. autoclass:: daal4py.pivoted_qr
   :members: compute
.. autoclass:: daal4py.pivoted_qr_result
   :members:

Normalization
-------------
Parameters and semantics are described in |onedal-dg-normalization|_.

Z-Score
^^^^^^^^
Parameters and semantics are described in |onedal-dg-z-score|_.

.. rubric:: Examples:

- `Single-Process Z-Score Normalization <https://github.com/IntelPython/daal4py/blob/master/examples/normalization_zscore_batch.py>`__

.. autoclass:: daal4py.normalization_zscore
   :members: compute
.. autoclass:: daal4py.normalization_zscore_result
   :members:

Min-Max
^^^^^^^^
Parameters and semantics are described in |onedal-dg-min-max|_.

.. rubric:: Examples:

- `Single-Process Min-Max Normalization <https://github.com/IntelPython/daal4py/blob/master/examples/normalization_minmax_batch.py>`__

.. autoclass:: daal4py.normalization_minmax
   :members: compute
.. autoclass:: daal4py.normalization_minmax_result
   :members:

Random Number Engines
---------------------
Parameters and semantics are described in |onedal-dg-engines|_.

.. autoclass:: daal4py.engines_result
   :members:

mt19937
^^^^^^^
Parameters and semantics are described in |onedal-dg-mt19937|_.

.. autoclass:: daal4py.engines_mt19937
   :members: compute
.. autoclass:: daal4py.engines_mt19937_result
   :members:

mt2203
^^^^^^^
Parameters and semantics are described in |onedal-dg-mt2203|_.

.. autoclass:: daal4py.engines_mt2203
   :members: compute
.. autoclass:: daal4py.engines_mt2203_result
   :members:

mcg59
^^^^^^^
Parameters and semantics are described in |onedal-dg-mcg59|_.

.. autoclass:: daal4py.engines_mcg59
   :members: compute
.. autoclass:: daal4py.engines_mcg59_result
   :members:

Distributions
-------------
Parameters and semantics are described in |onedal-dg-distributions|_.

Bernoulli
^^^^^^^^^
Parameters and semantics are described in |onedal-dg-bernoulli-distribution|_.

.. rubric:: Examples:

- `Single-Process Bernoulli Distribution <https://github.com/IntelPython/daal4py/blob/master/examples/distributions_bernoulli_batch.py>`__

.. autoclass:: daal4py.distributions_bernoulli
   :members: compute
.. autoclass:: daal4py.distributions_bernoulli_result
   :members:

Normal
^^^^^^
Parameters and semantics are described in |onedal-dg-normal-distribution|_.

.. rubric:: Examples:

- `Single-Process Normal Distribution <https://github.com/IntelPython/daal4py/blob/master/examples/distributions_normal_batch.py>`__

.. autoclass:: daal4py.distributions_normal
   :members: compute
.. autoclass:: daal4py.distributions_normal_result
   :members:

Uniform
^^^^^^^
Parameters and semantics are described in |onedal-dg-uniform-distribution|_.

.. rubric:: Examples:

- `Single-Process Uniform Distribution <https://github.com/IntelPython/daal4py/blob/master/examples/distributions_uniform_batch.py>`__

.. autoclass:: daal4py.distributions_uniform
   :members: compute
.. autoclass:: daal4py.distributions_uniform_result
   :members:

Association Rules
-----------------
Parameters and semantics are described in |onedal-dg-association-rules|_.

.. rubric:: Examples:

- `Single-Process Association Rules <https://github.com/IntelPython/daal4py/blob/master/examples/association_rules_batch.py>`__

.. autoclass:: daal4py.association_rules
   :members: compute
.. autoclass:: daal4py.association_rules_result
   :members:

Cholesky Decomposition
----------------------
Parameters and semantics are described in |onedal-dg-cholesky-decomposition|_.

.. rubric:: Examples:

- `Single-Process Cholesky <https://github.com/IntelPython/daal4py/blob/master/examples/cholesky_batch.py>`__

.. autoclass:: daal4py.cholesky
   :members: compute
.. autoclass:: daal4py.cholesky_result
   :members:

Correlation and Variance-Covariance Matrices
--------------------------------------------
Parameters and semantics are described in |onedal-dg-correlation-and-variance-covariance-matrices|_.

.. rubric:: Examples:

- `Single-Process Covariance <https://github.com/IntelPython/daal4py/blob/master/examples/covariance_batch.py>`__
- `Streaming Covariance <https://github.com/IntelPython/daal4py/blob/master/examples/covariance_streaming.py>`__
- `Multi-Process Covariance <https://github.com/IntelPython/daal4py/blob/master/examples/covariance_spmd.py>`__

.. autoclass:: daal4py.covariance
   :members: compute
.. autoclass:: daal4py.covariance_result
   :members:

Implicit Alternating Least Squares (implicit ALS)
-------------------------------------------------
Parameters and semantics are described in |onedal-dg-implicit-alternating-least-squares|_.

.. rubric:: Examples:

- `Single-Process implicit ALS <https://github.com/IntelPython/daal4py/blob/master/examples/implicit_als_batch.py>`__

.. autoclass:: daal4py.implicit_als_training
   :members: compute
.. autoclass:: daal4py.implicit_als_training_result
   :members:
.. autoclass:: daal4py.implicit_als_model
   :members:
.. autoclass:: daal4py.implicit_als_prediction_ratings
   :members: compute
.. autoclass:: daal4py.implicit_als_prediction_ratings_result
   :members:

Moments of Low Order
--------------------
Parameters and semantics are described in |onedal-dg-moments-of-low-order|_.

.. rubric:: Examples:

- `Single-Process Low Order Moments <https://github.com/IntelPython/daal4py/blob/master/examples/low_order_moms_dense_batch.py>`__
- `Streaming Low Order Moments <https://github.com/IntelPython/daal4py/blob/master/examples/low_order_moms_streaming.py>`__
- `Multi-Process Low Order Moments <https://github.com/IntelPython/daal4py/blob/master/examples/low_order_moms_spmd.py>`__

.. autoclass:: daal4py.low_order_moments
   :members: compute
.. autoclass:: daal4py.low_order_moments_result
   :members:

Quantiles
---------
Parameters and semantics are described in |onedal-dg-quantiles|_.

.. rubric:: Examples:

- `Single-Process Quantiles <https://github.com/IntelPython/daal4py/blob/master/examples/quantiles_batch.py>`__

.. autoclass:: daal4py.quantiles
   :members: compute
.. autoclass:: daal4py.quantiles_result
   :members:

Singular Value Decomposition (SVD)
----------------------------------
Parameters and semantics are described in |onedal-dg-svd|_.

.. rubric:: Examples:

- `Single-Process SVD <https://github.com/IntelPython/daal4py/blob/master/examples/svd_batch.py>`__
- `Streaming SVD <https://github.com/IntelPython/daal4py/blob/master/examples/svd_streaming.py>`__
- `Multi-Process SVD <https://github.com/IntelPython/daal4py/blob/master/examples/svd_spmd.py>`__

.. autoclass:: daal4py.svd
   :members: compute
.. autoclass:: daal4py.svd_result
   :members:

Sorting
-------
Parameters and semantics are described in |onedal-dg-sorting|_.

.. rubric:: Examples:

- `Single-Process Sorting <https://github.com/IntelPython/daal4py/blob/master/examples/sorting_batch.py>`__

.. autoclass:: daal4py.sorting
   :members: compute
.. autoclass:: daal4py.sorting_result
   :members:

Trees
-----
.. autofunction:: daal4py.getTreeState

.. rubric:: Examples:

- `Decision Forest Regression <https://github.com/IntelPython/daal4py/blob/master/examples/decision_forest_regression_traverse_batch.py>`__
- `Decision Forest Classification <https://github.com/IntelPython/daal4py/blob/master/examples/decision_forest_classification_traverse_batch.py>`__
- `Decision Tree Regression <https://github.com/IntelPython/daal4py/blob/master/examples/decision_tree_regression_traverse_batch.py>`__
- `Decision Tree Classification <https://github.com/IntelPython/daal4py/blob/master/examples/decision_tree_classification_traverse_batch.py>`__
- `Gradient Boosted Trees Regression <https://github.com/IntelPython/daal4py/blob/master/examples/gradient_boosted_regression_traverse_batch.py>`__
- `Gradient Boosted Trees Classification <https://github.com/IntelPython/daal4py/blob/master/examples/gradient_boosted_classification_traverse_batch.py>`__

.. Link replacements

.. |onedal-dg-bernoulli-distribution| replace:: Intel(R) oneAPI Data Analytics Library Bernoulli Distribution
.. _onedal-dg-bernoulli-distribution: https://oneapi-src.github.io/oneDAL/daal/algorithms/distributions/bernoulli.html

.. |onedal-dg-svd| replace:: Intel(R) oneAPI Data Analytics Library SVD
.. _onedal-dg-svd: https://oneapi-src.github.io/oneDAL/daal/algorithms/svd/singular-value-decomposition.html

.. |onedal-dg-regression| replace:: Intel(R) oneAPI Data Analytics Library Regression
.. _onedal-dg-regression: https://oneapi-src.github.io/oneDAL/daal/usage/training-and-prediction/regression.html

.. |onedal-dg-k-means-clustering| replace:: Intel(R) oneAPI Data Analytics Library K-Means Clustering
.. _onedal-dg-k-means-clustering: https://oneapi-src.github.io/oneDAL/daal/algorithms/kmeans/k-means-clustering.html

.. |onedal-dg-lbfgs| replace:: Intel(R) oneAPI Data Analytics Library LBFGS
.. _onedal-dg-lbfgs: https://oneapi-src.github.io/oneDAL/daal/algorithms/optimization-solvers/solvers/limited-memory-broyden-fletcher-goldfarb-shanno-algorithm.html

.. |onedal-dg-naive-bayes| replace:: Intel(R) oneAPI Data Analytics Library Naive Bayes
.. _onedal-dg-naive-bayes: https://oneapi-src.github.io/oneDAL/daal/algorithms/naive_bayes/naive-bayes-classifier.html

.. |onedal-dg-expectation-maximization| replace:: Intel(R) oneAPI Data Analytics Library Expectation-Maximization
.. _onedal-dg-expectation-maximization: https://oneapi-src.github.io/oneDAL/daal/algorithms/em/expectation-maximization.html

.. |onedal-dg-mcg59| replace:: Intel(R) oneAPI Data Analytics Library mcg59
.. _onedal-dg-mcg59: https://oneapi-src.github.io/oneDAL/daal/algorithms/engines/mcg59.html

.. |onedal-dg-least-absolute-shrinkage-and-selection-operator| replace:: Intel(R) oneAPI Data Analytics Library Least Absolute Shrinkage and Selection Operator
.. _onedal-dg-least-absolute-shrinkage-and-selection-operator: https://oneapi-src.github.io/oneDAL/daal/algorithms/lasso_elastic_net/lasso.html

.. |onedal-dg-sorting| replace:: Intel(R) oneAPI Data Analytics Library Sorting
.. _onedal-dg-sorting: https://oneapi-src.github.io/oneDAL/daal/algorithms/sorting/index.html

.. |onedal-dg-expectation-maximization-for-the-gaussian-mixture-model| replace:: Intel(R) oneAPI Data Analytics Library Expectation-Maximization for the Gaussian Mixture Model
.. _onedal-dg-expectation-maximization-for-the-gaussian-mixture-model: https://oneapi-src.github.io/oneDAL/daal/algorithms/em/expectation-maximization.html#em-algorithm-for-the-gaussian-mixture-model

.. |onedal-dg-multivariate-outlier-detection| replace:: Intel(R) oneAPI Data Analytics Library Multivariate Outlier Detection
.. _onedal-dg-multivariate-outlier-detection: https://oneapi-src.github.io/oneDAL/daal/algorithms/outlier_detection/multivariate.html

.. |onedal-dg-expectation-maximization-initialization| replace:: Intel(R) oneAPI Data Analytics Library Expectation-Maximization Initialization
.. _onedal-dg-expectation-maximization-initialization: https://oneapi-src.github.io/oneDAL/daal/algorithms/em/expectation-maximization.html#initialization

.. |onedal-dg-pivoted-qr-decomposition| replace:: Intel(R) oneAPI Data Analytics Library Pivoted QR Decomposition
.. _onedal-dg-pivoted-qr-decomposition: https://oneapi-src.github.io/oneDAL/daal/algorithms/qr/qr-pivoted.html

.. |onedal-dg-regression-decision-tree| replace:: Intel(R) oneAPI Data Analytics Library Regression Decision Tree
.. _onedal-dg-regression-decision-tree: https://oneapi-src.github.io/oneDAL/daal/algorithms/decision_tree/decision-tree-regression.html

.. |onedal-dg-k-nearest-neighbors-knn| replace:: Intel(R) oneAPI Data Analytics Library k-Nearest Neighbors (kNN)
.. _onedal-dg-k-nearest-neighbors-knn: https://oneapi-src.github.io/oneDAL/daal/algorithms/k_nearest_neighbors/k-nearest-neighbors-knn-classifier.html

.. |onedal-dg-pca| replace:: Intel(R) oneAPI Data Analytics Library PCA
.. _onedal-dg-pca: https://oneapi-src.github.io/oneDAL/daal/algorithms/pca/principal-component-analysis.html

.. |onedal-dg-sgd| replace:: Intel(R) oneAPI Data Analytics Library SGD
.. _onedal-dg-sgd: https://oneapi-src.github.io/oneDAL/daal/algorithms/optimization-solvers/solvers/stochastic-gradient-descent-algorithm.html

.. |onedal-dg-uniform-distribution| replace:: Intel(R) oneAPI Data Analytics Library Uniform Distribution
.. _onedal-dg-uniform-distribution: https://oneapi-src.github.io/oneDAL/daal/algorithms/distributions/uniform.html

.. |onedal-dg-cross-entropy-loss| replace:: Intel(R) oneAPI Data Analytics Library Cross Entropy Loss
.. _onedal-dg-cross-entropy-loss: https://oneapi-src.github.io/oneDAL/daal/algorithms/optimization-solvers/objective-functions/cross-entropy.html

.. |onedal-dg-classification| replace:: Intel(R) oneAPI Data Analytics Library Classification
.. _onedal-dg-classification: https://oneapi-src.github.io/oneDAL/daal/usage/training-and-prediction/classification.html

.. |onedal-dg-cosine-distance| replace:: Intel(R) oneAPI Data Analytics Library Cosine Distance
.. _onedal-dg-cosine-distance: https://oneapi-src.github.io/oneDAL/daal/algorithms/distance/cosine.html

.. |onedal-dg-regression-stump| replace:: Intel(R) oneAPI Data Analytics Library Regression Stump
.. _onedal-dg-regression-stump: https://oneapi-src.github.io/oneDAL/daal/algorithms/stump/regression.html

.. |onedal-dg-multivariate-bacon-outlier-detection| replace:: Intel(R) oneAPI Data Analytics Library Multivariate Bacon Outlier Detection
.. _onedal-dg-multivariate-bacon-outlier-detection: https://oneapi-src.github.io/oneDAL/daal/algorithms/outlier_detection/multivariate-bacon.html

.. |onedal-dg-logistic-regression| replace:: Intel(R) oneAPI Data Analytics Library Logistic Regression
.. _onedal-dg-logistic-regression: https://oneapi-src.github.io/oneDAL/daal/algorithms/logistic_regression/logistic-regression.html

.. |onedal-dg-quantiles| replace:: Intel(R) oneAPI Data Analytics Library Quantiles
.. _onedal-dg-quantiles: https://oneapi-src.github.io/oneDAL/daal/algorithms/quantiles/index.html

.. |onedal-dg-pca-transform| replace:: Intel(R) oneAPI Data Analytics Library PCA Transform
.. _onedal-dg-pca-transform: https://oneapi-src.github.io/oneDAL/daal/algorithms/pca/transform.html

.. |onedal-dg-correlation-distance| replace:: Intel(R) oneAPI Data Analytics Library Correlation Distance
.. _onedal-dg-correlation-distance: https://oneapi-src.github.io/oneDAL/daal/algorithms/distance/correlation.html

.. |onedal-dg-association-rules| replace:: Intel(R) oneAPI Data Analytics Library Association Rules
.. _onedal-dg-association-rules: https://oneapi-src.github.io/oneDAL/daal/algorithms/association_rules/association-rules.html

.. |onedal-dg-univariate-outlier-detection| replace:: Intel(R) oneAPI Data Analytics Library Univariate Outlier Detection
.. _onedal-dg-univariate-outlier-detection: https://oneapi-src.github.io/oneDAL/daal/algorithms/outlier_detection/univariate.html

.. |onedal-dg-classification-gradient-boosted-tree| replace:: Intel(R) oneAPI Data Analytics Library Classification Gradient Boosted Tree
.. _onedal-dg-classification-gradient-boosted-tree: https://oneapi-src.github.io/oneDAL/daal/algorithms/gradient_boosted_trees/gradient-boosted-trees-classification.html

.. |onedal-dg-classification-brownboost| replace:: Intel(R) oneAPI Data Analytics Library Classification BrownBoost
.. _onedal-dg-classification-brownboost: https://oneapi-src.github.io/oneDAL/daal/algorithms/boosting/brownboost.html

.. |onedal-dg-regression-decision-forest| replace:: Intel(R) oneAPI Data Analytics Library Regression Decision Forest
.. _onedal-dg-regression-decision-forest: https://oneapi-src.github.io/oneDAL/daal/algorithms/decision_forest/decision-forest-regression.html

.. |onedal-dg-z-score| replace:: Intel(R) oneAPI Data Analytics Library Z-Score
.. _onedal-dg-z-score: https://oneapi-src.github.io/oneDAL/daal/algorithms/normalization/z-score.html

.. |onedal-dg-classification-weak-learner-stump| replace:: Intel(R) oneAPI Data Analytics Library Classification Weak Learner Stump
.. _onedal-dg-classification-weak-learner-stump: https://oneapi-src.github.io/oneDAL/daal/algorithms/stump/classification.html

.. |onedal-dg-svm| replace:: Intel(R) oneAPI Data Analytics Library SVM
.. _onedal-dg-svm: https://oneapi-src.github.io/oneDAL/daal/algorithms/svm/support-vector-machine-classifier.html

.. |onedal-dg-regression-gradient-boosted-tree| replace:: Intel(R) oneAPI Data Analytics Library Regression Gradient Boosted Tree
.. _onedal-dg-regression-gradient-boosted-tree: https://oneapi-src.github.io/oneDAL/daal/algorithms/gradient_boosted_trees/gradient-boosted-trees-regression.html

.. |onedal-dg-logistic-loss| replace:: Intel(R) oneAPI Data Analytics Library Logistic Loss
.. _onedal-dg-logistic-loss: https://oneapi-src.github.io/oneDAL/daal/algorithms/optimization-solvers/objective-functions/logistic-loss.html

.. |onedal-dg-adagrad| replace:: Intel(R) oneAPI Data Analytics Library AdaGrad
.. _onedal-dg-adagrad: https://oneapi-src.github.io/oneDAL/daal/algorithms/optimization-solvers/solvers/adaptive-subgradient-method.html

.. |onedal-dg-qr-decomposition| replace:: Intel(R) oneAPI Data Analytics Library QR Decomposition
.. _onedal-dg-qr-decomposition: https://oneapi-src.github.io/oneDAL/daal/algorithms/qr/qr-decomposition.html

.. |onedal-dg-mt19937| replace:: Intel(R) oneAPI Data Analytics Library mt19937
.. _onedal-dg-mt19937: https://oneapi-src.github.io/oneDAL/daal/algorithms/engines/mt19937.html

.. |onedal-dg-implicit-alternating-least-squares| replace:: Intel(R) oneAPI Data Analytics Library Implicit Alternating Least Squares
.. _onedal-dg-implicit-alternating-least-squares: https://oneapi-src.github.io/oneDAL/daal/algorithms/implicit_als/implicit-alternating-least-squares.html

.. |onedal-dg-linear-regression| replace:: Intel(R) oneAPI Data Analytics Library Linear Regression
.. _onedal-dg-linear-regression: https://oneapi-src.github.io/oneDAL/daal/algorithms/linear_ridge_regression/linear-regression.html

.. |onedal-dg-classification-adaboost| replace:: Intel(R) oneAPI Data Analytics Library Classification AdaBoost
.. _onedal-dg-classification-adaboost: https://oneapi-src.github.io/oneDAL/daal/algorithms/boosting/adaboost.html

.. |onedal-dg-distributions| replace:: Intel(R) oneAPI Data Analytics Library Distributions
.. _onedal-dg-distributions: https://oneapi-src.github.io/oneDAL/daal/algorithms/distributions/index.html

.. |onedal-dg-correlation-and-variance-covariance-matrices| replace:: Intel(R) oneAPI Data Analytics Library Correlation and Variance-Covariance Matrices
.. _onedal-dg-correlation-and-variance-covariance-matrices: https://oneapi-src.github.io/oneDAL/daal/algorithms/covariance/correlation-and-variance-covariance-matrices.html

.. |onedal-dg-classification-decision-tree| replace:: Intel(R) oneAPI Data Analytics Library Classification Decision Tree
.. _onedal-dg-classification-decision-tree: https://oneapi-src.github.io/oneDAL/daal/algorithms/decision_tree/decision-tree-classification.html

.. |onedal-dg-ridge-regression| replace:: Intel(R) oneAPI Data Analytics Library Ridge Regression
.. _onedal-dg-ridge-regression: https://oneapi-src.github.io/oneDAL/daal/algorithms/linear_ridge_regression/ridge-regression.html

.. |onedal-dg-classification-logitboost| replace:: Intel(R) oneAPI Data Analytics Library Classification LogitBoost
.. _onedal-dg-classification-logitboost: https://oneapi-src.github.io/oneDAL/daal/algorithms/boosting/logitboost.html

.. |onedal-dg-k-means-initialization| replace:: Intel(R) oneAPI Data Analytics Library K-Means Initialization
.. _onedal-dg-k-means-initialization: https://oneapi-src.github.io/oneDAL/daal/algorithms/kmeans/k-means-clustering.html#initialization

.. |onedal-dg-qr-decomposition-without-pivoting| replace:: Intel(R) oneAPI Data Analytics Library QR Decomposition without pivoting
.. _onedal-dg-qr-decomposition-without-pivoting: https://oneapi-src.github.io/oneDAL/daal/algorithms/qr/qr-without-pivoting.html

.. |onedal-dg-mse| replace:: Intel(R) oneAPI Data Analytics Library MSE
.. _onedal-dg-mse: https://oneapi-src.github.io/oneDAL/daal/algorithms/optimization-solvers/objective-functions/mse.html

.. |onedal-dg-stochastic-average-gradient-descent-saga| replace:: Intel(R) oneAPI Data Analytics Library Stochastic Average Gradient Descent SAGA
.. _onedal-dg-stochastic-average-gradient-descent-saga: https://oneapi-src.github.io/oneDAL/daal/algorithms/optimization-solvers/solvers/stochastic-average-gradient-accelerated-method.html

.. |onedal-dg-engines| replace:: Intel(R) oneAPI Data Analytics Library Engines
.. _onedal-dg-engines: https://oneapi-src.github.io/oneDAL/daal/algorithms/engines/index.html

.. |onedal-dg-cholesky-decomposition| replace:: Intel(R) oneAPI Data Analytics Library Cholesky Decomposition
.. _onedal-dg-cholesky-decomposition: https://oneapi-src.github.io/oneDAL/daal/algorithms/cholesky/cholesky.html

.. |onedal-dg-classification-decision-forest| replace:: Intel(R) oneAPI Data Analytics Library Classification Decision Forest
.. _onedal-dg-classification-decision-forest: https://oneapi-src.github.io/oneDAL/daal/algorithms/decision_forest/decision-forest-classification.html

.. |onedal-dg-normalization| replace:: Intel(R) oneAPI Data Analytics Library Normalization
.. _onedal-dg-normalization: https://oneapi-src.github.io/oneDAL/daal/algorithms/normalization/index.html

.. |onedal-dg-density-based-spatial-clustering-of-applications-with-noise| replace:: Intel(R) oneAPI Data Analytics Library Density-Based Spatial Clustering of Applications with Noise
.. _onedal-dg-density-based-spatial-clustering-of-applications-with-noise: https://oneapi-src.github.io/oneDAL/daal/algorithms/dbscan/index.html

.. |onedal-dg-moments-of-low-order| replace:: Intel(R) oneAPI Data Analytics Library Moments of Low Order
.. _onedal-dg-moments-of-low-order: https://oneapi-src.github.io/oneDAL/daal/algorithms/moments/moments-of-low-order.html

.. |onedal-dg-mt2203| replace:: Intel(R) oneAPI Data Analytics Library mt2203
.. _onedal-dg-mt2203: https://oneapi-src.github.io/oneDAL/daal/algorithms/engines/mt2203.html

.. |onedal-dg-normal-distribution| replace:: Intel(R) oneAPI Data Analytics Library Normal Distribution
.. _onedal-dg-normal-distribution: https://oneapi-src.github.io/oneDAL/daal/algorithms/distributions/normal.html

.. |onedal-dg-k-means-computation| replace:: Intel(R) oneAPI Data Analytics Library K-Means Computation
.. _onedal-dg-k-means-computation: https://oneapi-src.github.io/oneDAL/daal/algorithms/kmeans/k-means-clustering.html#computation

.. |onedal-dg-min-max| replace:: Intel(R) oneAPI Data Analytics Library Min-Max
.. _onedal-dg-min-max: https://oneapi-src.github.io/oneDAL/daal/algorithms/normalization/min-max.html

