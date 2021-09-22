.. ******************************************************************************
.. * Copyright 2020-2021 Intel Corporation
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

.. _knn_mnist:

########################################################
Intel® Extension for Scikit-learn* KNN for MNIST dataset
########################################################

|

.. image:: ../_static/GitHub-Mark-32px.png
    :align: center
    :target: https://github.com/intel/scikit-learn-intelex/blob/master/examples/notebooks/knn_mnist.ipynb

.. centered:: `View source on GitHub <https://github.com/intel/scikit-learn-intelex/blob/master/examples/notebooks/knn_mnist.ipynb>`_

|

**In this guide we will describe how to scale out Scikit-learn KNN with Intel® Extension for Scikit-learn**

To demonstrate how KNN works, we took the MNIST dataset with 784 features:

.. code-block:: python
    :linenos:

    from time import time
    from sklearn import metrics
    from sklearn.model_selection import train_test_split
    
    from sklearn.datasets import fetch_openml
    x, y = fetch_openml(name='mnist_784', return_X_y=True)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=72)

Intel Extension for Scikit-learn (previously known as daal4py) contains drop-in replacement functionality
for the stock scikit-learn package. You can take advantage of the performance optimizations of
Intel Extension for Scikit-learn by adding just two lines of code before the usual scikit-learn imports:

.. code-block:: python
    :linenos:

    from sklearnex import patch_sklearn
    patch_sklearn()

.. code-block:: text

    Intel(R) Extension for Scikit-learn\* enabled (https://github.com/intel/scikit-learn-intelex)

Intel(R) Extension for Scikit-learn patching affects performance of specific Scikit-learn functionality.
Refer to the `list of supported algorithms and parameters <https://intel.github.io/scikit-learn-intelex/algorithms.html>`_ for details.
In cases when unsupported parameters are used, the package fallbacks into original Scikit-learn.
If the patching does not cover your scenarios, `submit an issue on GitHub <https://github.com/intel/scikit-learn-intelex/issues>`_.:

.. code-block:: python
    :linenos:

    params = {
        'n_neighbors': 40,
        'weights': 'distance',
        'n_jobs': -1
    }

    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(**params).fit(x_train, y_train)

    report = metrics.classification_report(y_test, predicted)
    print(f"Classification report for KNN:\n{report}\n")

.. code-block:: text

    Classification report for KNN:
                  precision    recall  f1-score   support

               0       0.97      0.99      0.98      1365
               1       0.93      0.99      0.96      1637
               2       0.99      0.94      0.96      1401
               3       0.96      0.95      0.96      1455
               4       0.98      0.96      0.97      1380
               5       0.95      0.95      0.95      1219
               6       0.96      0.99      0.97      1317
               7       0.94      0.95      0.95      1420
               8       0.99      0.90      0.94      1379
               9       0.92      0.94      0.93      1427

        accuracy                           0.96     14000
       macro avg       0.96      0.96      0.96     14000
    weighted avg       0.96      0.96      0.96     14000

With scikit-learn-intelex patching you can:

- Use your scikit-learn code for training and prediction with minimal changes (a couple of lines of code);
- Fast execution training and prediction of scikit-learn models;
- Get the same quality;
- Get speedup more than **24** times.


