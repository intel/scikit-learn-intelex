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

.. _get_started:

############################################
Quick Start
############################################

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

    import numpy as np
    from sklearnex import patch_sklearn
    patch_sklearn()

    # You need to re-import scikit-learn algorithms after the patch
    from sklearn.cluster import KMeans

    X = np.array([[1,  2], [1,  4], [1,  0],
                  [10, 2], [10, 4], [10, 0]])
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    print(f"kmeans.labels_ = {kmeans.labels_}")

In the example above, you can see that the use of the original Scikit-learn
has not changed. This behavior is achieved through drop-in patching.

To undo the patch, run::

    sklearnex.unpatch_sklearn()
    # You need to re-import scikit-learn algorithms after the unpatch:
    from sklearn.cluster import KMeans

You may specify which algorithms to patch:

- Patching only one algorithm::

    from sklearnex import patch_sklearn
    # The names match scikit-learn estimators
    patch_sklearn("SVC")

- Patching several algorithms::

    from sklearnex import patch_sklearn
    # The names match scikit-learn estimators
    patch_sklearn(["SVC", "DBSCAN"])

Intel(R) Extension for Scikit-learn does not patch all scikit-learn algorithms and parameters.
You can find the :ref:`full patching map here <sklearn_algorithms>`.

.. note::
    Intel(R) Extension for Scikit-learn supports optimizations for the last four versions of scikit-learn.
    The latest release of scikit-learn-intelex-2021.3.X supports scikit-learn 0.22.X, 0.23.X, 0.24.X and 1.0.X.

