.. ******************************************************************************
.. * Copyright 2024 Intel Corporation
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

Non-Scikit Algorithms
=====================
Algorithms not presented in the original Scikit-learn are described here. All algorithms are 
available for both CPU and GPU (including distributed mode)

BasicStatistics
---------------
Calculates basic statistics of the given data.

Parameters
**********
**result_options**: *{str, list}, default='all'*

Used to set statistics to calculate. Possible values are ``'min'``, ``'max'``, ``'sum'``, ``'mean'``, ``'variance'``,
``'variation'``, ``sum_squares'``, ``sum_squares_centered'``, ``'standard_deviation'``, ``'second_order_raw_moment'``
or a list containing any of these values. If set to ``'all'`` then all possible statistics will be 
calculated.

Attributes
**********
After call of ``fit()`` each result is saved in the attribute with corresponding result option name, i.e.
``sum_squares`` (deprecated in version 2025.0) or in an attribute named by result option name with trailing underscore,
i.e. ``sum_squares_``

Examples
********
.. code-block:: python

    >>> import numpy as np
    >>> from sklearnex.basic_statistics import BasicStatistics
    >>> bs = BasicStatistics(result_options=['sum', 'min', 'max'])
    >>> X = np.array([[1, 2], [3, 4]])
    >>> bs.fit(X)
    >>> bs.sum_ # np.array([4., 6.])
    >>> bs.min_ # np.array([1., 2.])
    



IncrementalEmpiricalCovariance
------------------------------
Maximum likelihood covariance estimator that allows for the estimation when the data are split into
batches. The user can use the ``partial_fit`` method to provide a single batch of data or use the ``fit`` method to provide
the entire dataset.

Parameters
**********
All parameters used in ``EmpiricalCovariance`` are valid here as well. The additional parameters are listed below.

**batch_size**: *{int}, default=5*

Size of data batches used in the ``fit`` method.
 
Attributes
**********
The same as for ``EmpiricalCovariance``.

Examples
********
.. code-block:: python

    >>> import numpy as np
    >>> from sklearnex.covariance import IncrementalEmpiricalCovariance
    >>> inccov = IncrementalEmpiricalCovariance(batch_size=1)
    >>> X = np.array([[1, 2], [3, 4]])
    >>> inccov.partial_fit(X[:1])
    >>> inccov.partial_fit(X[1:])
    >>> inccov.covariance_ # np.array([[1., 1.],[1., 1.]])
    >>> inccov.location_ # np.array([2., 3.])
    >>> inccov.fit(X)
    >>> inccov.covariance_ # np.array([[1., 1.],[1., 1.]])
    >>> inccov.location_ # np.array([2., 3.])

IncrementalBasicStatistics
--------------------------
Calculates basic statistics on the given data, allows for computation when the data are split into
batches. The user can use ``partial_fit`` method to provide a single batch of data or use the ``fit`` method to provide
the entire dataset.

Parameters
**********
All parameters used in ``BasicStatistics`` are valid here as well. The additional parameters are listed below.

**batch_size**: *{int}, default=5*

Size of data batches used in the ``fit`` method.

Attributes
**********
The same as for ``BasicStatistics``.

Examples
********
.. code-block:: python

    >>> import numpy as np
    >>> from sklearnex.basic_statistics import IncrementalBasicStatistics
    >>> incbs = IncrementalBasicStatistics(batch_size=1)
    >>> X = np.array([[1, 2], [3, 4]])
    >>> incbs.partial_fit(X[:1])
    >>> incbs.partial_fit(X[1:])
    >>> incbs.sum_ # np.array([4., 6.])
    >>> incbs.min_ # np.array([1., 2.])
    >>> incbs.fit(X)
    >>> incbs.sum_ # np.array([4., 6.])
    >>> incbs.max_ # np.array([3., 4.])

IncrementalLinearRegression
---------------------------
Trains a linear regression model, allows for computation if the data are split into
batches. The user can use the ``partial_fit`` method to provide a single batch of data or use the ``fit`` method to provide
the entire dataset.

Parameters
**********
All parameters used in ``LinearRegression`` are valid here as well. The additional parameters are listed below.

**batch_size**: *{int}, default=5*

Size of data batches used in ``fit`` method.

Attributes
**********
The same as for ``LinearRegression``.

Examples
********
.. code-block:: python

    >>> import numpy as np
    >>> from sklearnex.linear_model import IncrementalLinearRegression
    >>> inclr = IncrementalLinearRegression(batch_size=2)
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 10]])
    >>> y = np.array([1.5, 3.5, 5.5, 8.5])
    >>> inclr.partial_fit(X[:2], y[:2])
    >>> inclr.partial_fit(X[2:], y[2:])
    >>> inclr.coef_ # np.array([0.5., 0.5.])
    >>> inclr.intercept_ # np.array(0.)
    >>> inclr.fit(X)
    >>> inclr.coef_ # np.array([0.5., 0.5.])
    >>> inclr.intercept_ # np.array(0.)
