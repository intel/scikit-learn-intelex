.. _index:

#####################################################
Intel(R) Extension for Scikit-learn*
#####################################################
Intel(R) Extension for Scikit-learn speeds up scikit-learn by providing drop-in patching.
Acceleration is achieved through the use of the Intel(R) oneAPI Data Analytics Library (oneDAL)
that allows for fast usage of the framework suited for Data Scientists or Machine Learning users.

Designed for Data Scientists and Framework Designers
----------------------------------------------------
Intel(R) Extension for Scikit-learn* was created to give data scientists the easiest way to get a better performance
while using the familiar scikit-learn package.

Usage
--------------------
Intel(R) Extension for Scikit-learn* dynamically patches scikit-learn estimators to use Intel(R) oneAPI Data Analytics Library
as the underlying solver, while getting the same solution faster.

- It is possible to enable those patches without editing the code of a scikit-learn application by
  using the following commandline flag::

    python -m sklearnex my_application.py

- Or from your script::

    from sklearnex import patch_sklearn
    patch_sklearn()


For example::

    from sklearnex import patch_sklearn
    patch_sklearn()

    from sklearn.cluster import KMeans
    import numpy as np
    X = np.array([[1, 2], [1, 4], [1, 0],
                  [10, 2], [10, 4], [10, 0]])
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    print(f"kmeans.labels_ = {kmeans.labels_}")
    pred = kmeans.predict([[0, 0], [12, 3]])
    print(f"pred = {pred}")
    print(f"kmeans.cluster_centers_ = {kmeans.cluster_centers_}")

In the example above, you can see that the use of the original Scikit-learn
has not changed. This behavior is achieved through drop-in patching.

To undo the patch Scikit-learn with::

    sklearnex.unpatch_sklearn()

Intel (R) Extension for Scikit-learn* does not patch all scikit-learn algorithms and parameters.
You can find the :ref:`full patching map here <sklearn_algorithms>`.

.. note::
    Intel(R) Extension for Scikit-learn* supports optimizations for the last four versions of scikit-learn.
    The latest release of scikit-learn-intelex-2021.2.X supports scikit-learn 0.21.X, 0.22.X, 0.23.X and 0.24.X.

Follow us on Medium
--------------------
We publish blogs on Medium, so `follow us <https://medium.com/intel-analytics-software/tagged/machine-learning>`_
to learn tips and tricks for more efficient data analysis the help of Intel(R) Extension for Scikit-learn.
Here are our latest blogs:

- `Intel Gives Scikit-Learn the Performance Boost Data Scientists Need <https://medium.com/intel-analytics-software/intel-gives-scikit-learn-the-performance-boost-data-scientists-need-42eb47c80b18>`_,
- `From Hours to Minutes: 600x Faster SVM <https://medium.com/intel-analytics-software/from-hours-to-minutes-600x-faster-svm-647f904c31ae>`_,
- `Improve the Performance of XGBoost and LightGBM Inference <https://medium.com/intel-analytics-software/improving-the-performance-of-xgboost-and-lightgbm-inference-3b542c03447e>`_,
- `Accelerate Kaggle Challenges Using Intel AI Analytics Toolkit <https://medium.com/intel-analytics-software/accelerate-kaggle-challenges-using-intel-ai-analytics-toolkit-beb148f66d5a>`_,
- `Accelerate Your scikit-learn Applications <https://medium.com/intel-analytics-software/improving-the-performance-of-xgboost-and-lightgbm-inference-3b542c03447e>`_,
- `Accelerate Linear Models for Machine Learning <https://medium.com/intel-analytics-software/accelerating-linear-models-for-machine-learning-5a75ff50a0fe>`_,
- `Accelerate K-Means Clustering <https://medium.com/intel-analytics-software/accelerate-k-means-clustering-6385088788a1>`_.

Important links
--------------------
- `Building from Sources <https://github.com/intel/scikit-learn-intelex/blob/master/INSTALL.md>`_,
- `About Intel(R) oneAPI Data Analytics Library <https://github.com/oneapi-src/oneDAL>`_.

Support
--------------------
Report issues, ask questions, and provide suggestions using:

- `GitHub Issues <https://github.com/intel/scikit-learn-intelex/issues>`_,
- `GitHub Discussions <https://github.com/intel/scikit-learn-intelex/discussions>`_,
- `Forum <https://community.intel.com/t5/Intel-Distribution-for-Python/bd-p/distribution-python>`_.

You may reach out to project maintainers privately at onedal.maintainers@intel.com
