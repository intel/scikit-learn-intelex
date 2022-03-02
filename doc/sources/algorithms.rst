.. ******************************************************************************
.. * Copyright 2020 Intel Corporation
.. *
.. * Licensed under the Apache License, Version 2.0 (the "License");
.. * you may not use this file except in compliance with the License.
.. * You may obtain a copy of the License at
.. *
.. *     http://www.apache.org/licenses/LICENSE-2.0
.. *
.. * Unless required by applicable law or agreed to in writing, software
.. * distributed under the License is distributed on an "AS IS" BASIS,
.. * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. * See the License for the specific language governing permissions and
.. * limitations under the License.
.. *******************************************************************************/

.. _sklearn_algorithms:

####################
Supported Algorithms
####################

Applying |intelex| will impact the following scikit-learn algorithms:

on CPU
------

.. list-table::
   :widths: 10 10 30 15
   :header-rows: 1
   :align: left

   * - Task
     - Functionality
     - Parameters support
     - Data support
   * - Classification
     - SVC
     - All parameters are supported
     - No limitations.
   * - Classification
     - NuSVC
     - All parameters are supported
     - No limitations.
   * - Classification
     - RandomForestClassifier
     - All parameters except ``warm_start`` = True, ``cpp_alpha`` != 0, ``criterion`` != 'gini'.
     - Multi-output and sparse data are not supported.
   * - Classification
     - KNeighborsClassifier
     - 
       - For ``algorithm`` == 'kd_tree':
           | all parameters except ``metric`` != 'euclidean' or 'minkowski' with ``p`` != 2.
       - For ``algorithm`` == 'brute':
           | all parameters except ``metric`` not in ['euclidean', 'manhattan', 'minkowski', 'chebyshev', 'cosine'].
     - Multi-output and sparse data is not supported.
   * - Classification
     - LogisticRegression
     - All parameters except ``solver`` != 'lbfgs' or 'newton-cg', ``class_weight`` != None, ``sample_weight`` != None.
     - Only dense data is supported.
   * - Regression
     - SVR
     - All parameters are supported
     - No limitations.
   * - Regression
     - NuSVR
     - All parameters are supported
     - No limitations.
   * - Regression
     - RandomForestRegressor
     - All parameters except ``warm_start`` = True, ``cpp_alpha`` != 0, ``criterion`` != 'mse'.
     - Multi-output and sparse data are not supported.
   * - Regression
     - KNeighborsRegressor
     - All parameters except ``metric`` != 'euclidean' or 'minkowski' with ``p`` != 2.
     - Multi-output and sparse data is not supported.
   * - Regression
     - LinearRegression
     - All parameters except ``normalize`` != False and ``sample_weight`` != None.
     - Only dense data is supported, #observations should be >= #features.
   * - Regression
     - Ridge
     - All parameters except ``normalize`` != False, ``solver`` != 'auto' and ``sample_weight`` != None.
     - Only dense data is supported, #observations should be >= #features.
   * - Regression
     - ElasticNet
     - All parameters except ``sample_weight`` != None.
     - Multi-output and sparse data is not supported, #observations should be >= #features.
   * - Regression
     - Lasso
     - All parameters except ``sample_weight`` != None.
     - Multi-output and sparse data is not supported, #observations should be >= #features.
   * - Clustering
     - KMeans
     - All parameters except ``precompute_distances`` and ``sample_weight`` != None.
     - No limitations.
   * - Clustering
     - DBSCAN
     - All parameters except ``metric`` != 'euclidean' or 'minkowski' with ``p`` != 2, ``algorithm`` != 'brute' or 'auto'.
     - Only dense data is supported.
   * - Dimensionality reduction
     - PCA
     - All parameters except ``svd_solver`` != 'full'.
     - Sparse data is not supported.
   * - Dimensionality reduction
     - TSNE
     - All parameters except ``metric`` != 'euclidean' or 'minkowski' with ``p`` != 2.
     - Sparse data is not supported.
   * - Unsupervised
     - NearestNeighbors
     - 
       - For ``algorithm`` == 'kd_tree':
           | all parameters except ``metric`` != 'euclidean' or 'minkowski' with ``p`` != 2.
       - For ``algorithm`` == 'brute':
           | all parameters except ``metric`` not in ['euclidean', 'manhattan', 'minkowski', 'chebyshev', 'cosine'].
     - Sparse data is not supported.
   * - Other
     - train_test_split
     - All parameters are supported.
     - Only dense data is supported.
   * - Other
     - assert_all_finite
     - All parameters are supported.
     - Only dense data is supported.
   * - Other
     - pairwise_distance
     - With ``metric`` = 'cosine' or 'correlation'.
     - Only dense data is supported.
   * - Other
     - roc_auc_score
     - Parameters ``average``, ``sample_weight``, ``max_fpr`` and ``multi_class`` are not supported.
     - No limitations.

on GPU
------

.. list-table::
   :widths: 10 10 30 15
   :header-rows: 1
   :align: left

   * - Task
     - Functionality
     - Parameters support
     - Data support
   * - Classification
     - SVC
     - All parameters except ``kernel`` = 'sigmoid_poly', ``class_weight`` != None.
     - Only binary dense data is supported.
   * - Classification
     - RandomForestClassifier
     - All parameters except ``warm_start`` = True, ``cpp_alpha`` != 0, ``criterion`` != 'gini', ``oob_score`` = True.
     - Multi-output, sparse data, out-of-bag score and sample_weight are not supported.
   * - Classification
     - KNeighborsClassifier
     - All parameters except ``algorithm`` != 'brute', ``weights`` = 'callable', ``metric`` not in ['euclidean', 'manhattan', 'minkowski', 'chebyshev', 'cosine'].
     - Only dense data is supported.
   * - Classification
     - LogisticRegression
     - All parameters except ``solver`` != 'newton-cg', ``class_weight`` != None, ``sample_weight`` != None, ``penalty`` != 'l2'
     - Only dense data is supported.
   * - Regression
     - RandomForestRegressor
     - All parameters except ``warm_start`` = True, ``cpp_alpha`` != 0, ``criterion`` != 'mse', ``oob_score`` = True.
     - Multi-output, sparse data, out-of-bag score and sample_weight are not supported.
   * - Regression
     - KNeighborsRegressor
     - All parameters except ``algorithm`` != 'brute', ``weights`` = 'callable', ``metric`` != 'euclidean' or 'minkowski' with ``p`` != 2.
     - Only dense data is supported.
   * - Regression
     - LinearRegression
     - All parameters except ``normalize`` != False and ``sample_weight`` != None.
     - Only dense data is supported, #observations should be >= #features.
   * - Clustering
     - KMeans
     - All parameters except ``precompute_distances`` and ``sample_weight`` != None. Init = 'k-means++' fallbacks to CPU.
     - Sparse data is not supported.
   * - Clustering
     - DBSCAN
     - All parameters except ``metric`` != 'euclidean', ``algorithm`` != 'brute', ``algorithm`` != 'auto'.
     - Only dense data is supported.
   * - Dimensionality reduction
     - PCA
     - All parameters except ``svd_solver`` != 'full'.
     - Sparse data is not supported.
   * - Unsupervised
     - NearestNeighbors
     - All parameters except ``algorithm`` != 'brute', ``weights`` = 'callable', ``metric`` not in ['euclidean', 'manhattan', 'minkowski', 'chebyshev', 'cosine'].
     - Only dense data is supported.

.. seealso:: :ref:`oneapi_gpu`

Scikit-learn tests
------------------

Monkey-patched scikit-learn classes and functions passes scikit-learn's own test
suite, with few exceptions, specified in `deselected_tests.yaml
<https://github.com/intel/scikit-learn-intelex/blob/master/deselected_tests.yaml>`__.

The results of the entire latest scikit-learn test suite with |intelex|: `CircleCI
<https://circleci.com/gh/intel/scikit-learn-intelex>`_.