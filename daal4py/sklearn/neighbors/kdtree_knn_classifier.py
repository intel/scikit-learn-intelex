# *******************************************************************************
# Copyright 2014-2019 Intel Corporation
# All Rights Reserved.
#
# This software is licensed under the Apache License, Version 2.0 (the
# "License"), the following terms apply:
#
# You may not use this file except in compliance with the License.  You may
# obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# See the License for the specific language governing permissions and
# limitations under the License.
# *******************************************************************************

# daal4py KNN scikit-learn-compatible estimator class

import numpy as np
import numbers
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn import preprocessing
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils import check_random_state
import daal4py as d4p
from ..utils import getFPType


class KNeighborsClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 n_neighbors=5,
                 weights='uniform',
                 algorithm='kd_tree',
                 leaf_size=31,
                 p=2,
                 metric='minkowski',
                 metric_params=None,
                 n_jobs=None):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs

    def fit(self, X, y):
        # Check the algorithm parameters
        if not ((isinstance(self.n_neighbors, numbers.Integral))
                and (self.n_neighbors > 0)):
            raise ValueError('Parameter "n_neighbors" must be '
                             'non-zero positive integer value.')
        if not self.weights == 'uniform':
            warnings.warn('Value "{}" for argument "weights" not supported. '
                          'Using default "uniform".'.format(self.weights),
                          RuntimeWarning, stacklevel=2)
            self.weights = 'uniform'
        if not self.algorithm == 'kd_tree':
            warnings.warn('Value "{}" for argument "algorithm" not supported. '
                          'Using default "kd_tree".'.format(self.algorithm),
                          RuntimeWarning, stacklevel=2)
            self.algorithm = 'kd_tree'
        if not self.leaf_size == 31:
            warnings.warn('Value "{}" for argument "leaf_size" not supported. '
                          'Using default "31".'.format(self.leaf_size),
                          RuntimeWarning, stacklevel=2)
            self.leaf_size = 31
        if not self.p == 2:
            warnings.warn('Value "{}" for argument "p" not supported. '
                          'Using default "2".'.format(self.p),
                          RuntimeWarning, stacklevel=2)
            self.p = 2
        if not self.metric == 'minkowski':
            warnings.warn('Value "{}" for argument "metric" not supported. '
                          'Using default "minkowski".'.format(self.metric),
                          RuntimeWarning, stacklevel=2)
            self.metric = 'minkowski'
        if self.metric_params is not None:
            warnings.warn('Argument "metric_params" not (yet) supported. '
                          'Ignored.',
                          RuntimeWarning, stacklevel=2)
            self.metric_params = None
        if self.n_jobs is not None:
            warnings.warn('Argument "n_jobs" not (yet) supported. '
                          'Ignored. All available processors will be used.',
                          RuntimeWarning, stacklevel=2)
            self.n_jobs = None

        # Check that X and y have correct shape
        X, y = check_X_y(X, y, y_numeric=False, dtype=[np.float64, np.float32])

        check_classification_targets(y)

        # Encode labels
        le = preprocessing.LabelEncoder()
        le.fit(y)
        self.classes_ = le.classes_
        y_ = le.transform(y)

        # Convert to 2d array
        y_ = y_.reshape((-1, 1))

        self.n_classes_ = len(self.classes_)

        self.n_features_ = X.shape[1]

        # Classifier can't train when only one class is present.
        # Trivial case
        if self.n_classes_ == 1:
            return self

        # Get random seed
        rs = check_random_state(None)
        self.seed_ = rs.randint(np.iinfo('i').max)

        # Define type of data
        fptype = getFPType(X)

        # Fit the model
        train_algo = d4p.kdtree_knn_classification_training(fptype=fptype,
                                                            engine=d4p.engines_mcg59(seed=self.seed_))
        train_result = train_algo.compute(X, y_)

        # Store the model
        self.daal_model_ = train_result.model

        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['n_features_',
                               'n_classes_'])

        # Input validation
        X = check_array(X)
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen in `fit`')

        # Trivial case
        if self.n_classes_ == 1:
            return np.full(X.shape[0], self.classes_[0])

        check_is_fitted(self, ['daal_model_'])

        # Define type of data
        fptype = getFPType(X)

        # Prediction
        predict_algo = d4p.kdtree_knn_classification_prediction(fptype=fptype,
                                                                k=self.n_neighbors)
        predict_result = predict_algo.compute(X, self.daal_model_)

        # Decode labels
        le = preprocessing.LabelEncoder()
        le.classes_ = self.classes_
        return le.inverse_transform(predict_result.prediction.ravel().astype(np.int64, copy=False))
