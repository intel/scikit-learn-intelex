# *******************************************************************************
# Copyright 2014-2020 Intel Corporation
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

# daal4py GBT scikit-learn-compatible estimator class

import numpy as np
import numbers
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn import preprocessing
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils import check_random_state
import daal4py as d4p
from ..utils import getFPType

from sklearn import __version__ as sklearn_version
from distutils.version import LooseVersion


class GBTDAALClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 splitMethod='inexact',
                 maxIterations=50,
                 maxTreeDepth=6,
                 shrinkage=0.3,
                 minSplitLoss=0,
                 lambda_=1,
                 observationsPerTreeFraction=1,
                 featuresPerNode=0,
                 minObservationsInLeafNode=5,
                 memorySavingMode=False,
                 maxBins=256,
                 minBinSize=5,
                 random_state=None):
        self.splitMethod = splitMethod
        self.maxIterations = maxIterations
        self.maxTreeDepth = maxTreeDepth
        self.shrinkage = shrinkage
        self.minSplitLoss = minSplitLoss
        self.lambda_ = lambda_
        self.observationsPerTreeFraction = observationsPerTreeFraction
        self.featuresPerNode = featuresPerNode
        self.minObservationsInLeafNode = minObservationsInLeafNode
        self.memorySavingMode = memorySavingMode
        self.maxBins = maxBins
        self.minBinSize = minBinSize
        self.random_state = random_state

    def fit(self, X, y):
        # Check the algorithm parameters
        if not self.splitMethod in ('inexact', 'exact'):
            warnings.warn('Value "{}" for argument "weights" not supported. '
                          'Using default "uniform".'.format(self.splitMethod),
                          RuntimeWarning, stacklevel=2)
        if not ((isinstance(self.maxIterations, numbers.Integral))
                and (self.maxIterations > 0)):
            raise ValueError('Parameter "maxIterations" must be '
                             'non-zero positive integer value.')
        if not ((isinstance(self.maxTreeDepth, numbers.Integral))
                and (self.maxTreeDepth >= 0)):
            raise ValueError('Parameter "maxTreeDepth" must be '
                             'positive integer value or zero.')
        if not ((self.shrinkage >= 0)
                and (self.shrinkage < 1)):
            raise ValueError('Parameter "shrinkage" must be '
                             'more or equal to 0 and less than 1.')
        if not (self.minSplitLoss >= 0):
            raise ValueError('Parameter "minSplitLoss" must be '
                             'more or equal to zero.')
        if not (self.lambda_ >= 0):
            raise ValueError('Parameter "lambda_" must be '
                             'more or equal to zero.')
        if not ((self.observationsPerTreeFraction > 0)
                and (self.observationsPerTreeFraction <= 1)):
            raise ValueError('Parameter "observationsPerTreeFraction" must be '
                             'more than 0 and less or equal to 1.')
        if not ((isinstance(self.featuresPerNode, numbers.Integral))
                and (self.featuresPerNode >= 0)):
            raise ValueError('Parameter "featuresPerNode" must be '
                             'positive integer value or zero.')
        if not ((isinstance(self.minObservationsInLeafNode, numbers.Integral))
                and (self.minObservationsInLeafNode > 0)):
            raise ValueError('Parameter "minObservationsInLeafNode" must be '
                             'non-zero positive integer value.')
        if not (isinstance(self.memorySavingMode, bool)):
            raise ValueError('Parameter "memorySavingMode" must be '
                             'boolean value.')
        if not ((isinstance(self.maxBins, numbers.Integral))
                and (self.maxBins > 0)):
            raise ValueError('Parameter "maxBins" must be '
                             'non-zero positive integer value.')
        if not ((isinstance(self.minBinSize, numbers.Integral))
                and (self.minBinSize > 0)):
            raise ValueError('Parameter "minBinSize" must be '
                             'non-zero positive integer value.')

        # Check that X and y have correct shape
        X, y = check_X_y(X, y, y_numeric=False, dtype=[np.single, np.double])

        check_classification_targets(y)

        # Encode labels
        le = preprocessing.LabelEncoder()
        le.fit(y)
        self.classes_ = le.classes_
        y_ = le.transform(y)

        # Convert to 2d array
        y_ = y_.reshape((-1, 1))

        self.n_outputs_ = y_.shape[1]

        self.n_classes_ = len(self.classes_)

        self.n_features_ = X.shape[1]

        # Classifier can't train when only one class is present.
        # Trivial case
        if self.n_classes_ == 1:
            return self

        # Get random seed
        rs_ = check_random_state(self.random_state)
        seed_ = rs_.randint(0, np.iinfo('i').max)

        # Define type of data
        fptype = getFPType(X)

        # Fit the model
        train_algo = d4p.gbt_classification_training(fptype=fptype,
                                                     nClasses=self.n_classes_,
                                                     splitMethod=self.splitMethod,
                                                     maxIterations=self.maxIterations,
                                                     maxTreeDepth=self.maxTreeDepth,
                                                     shrinkage=self.shrinkage,
                                                     minSplitLoss=self.minSplitLoss,
                                                     lambda_=self.lambda_,
                                                     observationsPerTreeFraction=self.observationsPerTreeFraction,
                                                     featuresPerNode=self.featuresPerNode,
                                                     minObservationsInLeafNode=self.minObservationsInLeafNode,
                                                     memorySavingMode=self.memorySavingMode,
                                                     maxBins=self.maxBins,
                                                     minBinSize=self.minBinSize,
                                                     engine=d4p.engines_mcg59(seed=seed_))
        train_result = train_algo.compute(X, y_)

        # Store the model
        self.daal_model_ = train_result.model

        # Return the classifier
        return self

    def _predict(self, X, resultsToEvaluate):
        # Check is fit had been called
        check_is_fitted(self, ['n_features_', 'n_classes_'])

        # Input validation
        X = check_array(X, dtype=[np.single, np.double])
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen in `fit`')

        # Trivial case
        if self.n_classes_ == 1:
            return np.full(X.shape[0], self.classes_[0])

        if not hasattr(self, 'daal_model_'):
            raise ValueError(("The class {} instance does not have 'daal_model_' attribute set. "
                              "Call 'fit' with appropriate arguments before using this method.").format(type(self).__name__))

        # Define type of data
        fptype = getFPType(X)

        # Prediction
        predict_algo = d4p.gbt_classification_prediction(fptype=fptype,
                                                         nClasses=self.n_classes_,
                                                         resultsToEvaluate=resultsToEvaluate)
        predict_result = predict_algo.compute(X, self.daal_model_)

        if resultsToEvaluate == "computeClassLabels":
            # Decode labels
            le = preprocessing.LabelEncoder()
            le.classes_ = self.classes_
            return le.inverse_transform(predict_result.prediction.ravel().astype(np.int64, copy=False))
        else:
            return predict_result.probabilities

    def predict(self, X):
        return self._predict(X, "computeClassLabels")

    def predict_proba(self, X):
        return self._predict(X, "computeClassProbabilities")

    def predict_log_proba(self, X):
        proba = self.predict_proba(X)

        if self.n_outputs_ == 1:
            return np.log(proba)
        else:
            for k in range(self.n_outputs_):
                proba[k] = np.log(proba[k])

            return proba

class GBTDAALRegressor(BaseEstimator, RegressorMixin):
    def __init__(self,
                 splitMethod='inexact',
                 maxIterations=2,
                 maxTreeDepth=2,
                 shrinkage=0.3,
                 minSplitLoss=0,
                 lambda_=1,
                 observationsPerTreeFraction=1,
                 featuresPerNode=0,
                 minObservationsInLeafNode=15,
                 memorySavingMode=False,
                 maxBins=256,
                 minBinSize=5,
                 random_state=None):
        self.splitMethod = splitMethod
        self.maxIterations = maxIterations
        self.maxTreeDepth = maxTreeDepth
        self.shrinkage = shrinkage
        self.minSplitLoss = minSplitLoss
        self.lambda_ = lambda_
        self.observationsPerTreeFraction = observationsPerTreeFraction
        self.featuresPerNode = featuresPerNode
        self.minObservationsInLeafNode = minObservationsInLeafNode
        self.memorySavingMode = memorySavingMode
        self.maxBins = maxBins
        self.minBinSize = minBinSize
        self.random_state = random_state

    def fit(self, X, y):
        # Check the algorithm parameters
        if not self.splitMethod in ('inexact', 'exact'):
            warnings.warn('Value "{}" for argument "weights" not supported. '
                          'Using default "uniform".'.format(self.splitMethod),
                          RuntimeWarning, stacklevel=2)
        if not ((isinstance(self.maxIterations, numbers.Integral))
                and (self.maxIterations > 0)):
            raise ValueError('Parameter "maxIterations" must be '
                             'non-zero positive integer value.')
        if not ((isinstance(self.maxTreeDepth, numbers.Integral))
                and (self.maxTreeDepth >= 0)):
            raise ValueError('Parameter "maxTreeDepth" must be '
                             'positive integer value or zero.')
        if not ((self.shrinkage >= 0)
                and (self.shrinkage < 1)):
            raise ValueError('Parameter "shrinkage" must be '
                             'more or equal to 0 and less than 1.')
        if not (self.minSplitLoss >= 0):
            raise ValueError('Parameter "minSplitLoss" must be '
                             'more or equal to zero.')
        if not (self.lambda_ >= 0):
            raise ValueError('Parameter "lambda_" must be '
                             'more or equal to zero.')
        if not ((self.observationsPerTreeFraction > 0)
                and (self.observationsPerTreeFraction <= 1)):
            raise ValueError('Parameter "observationsPerTreeFraction" must be '
                             'more than 0 and less or equal to 1.')
        if not ((isinstance(self.featuresPerNode, numbers.Integral))
                and (self.featuresPerNode >= 0)):
            raise ValueError('Parameter "featuresPerNode" must be '
                             'positive integer value or zero.')
        if not ((isinstance(self.minObservationsInLeafNode, numbers.Integral))
                and (self.minObservationsInLeafNode > 0)):
            raise ValueError('Parameter "minObservationsInLeafNode" must be '
                             'non-zero positive integer value.')
        if not (isinstance(self.memorySavingMode, bool)):
            raise ValueError('Parameter "memorySavingMode" must be '
                             'boolean value.')
        if not ((isinstance(self.maxBins, numbers.Integral))
                and (self.maxBins > 0)):
            raise ValueError('Parameter "maxBins" must be '
                             'non-zero positive integer value.')
        if not ((isinstance(self.minBinSize, numbers.Integral))
                and (self.minBinSize > 0)):
            raise ValueError('Parameter "minBinSize" must be '
                             'non-zero positive integer value.')

        # Check that X and y have correct shape
        X, y = check_X_y(X, y, y_numeric=True, dtype=[np.single, np.double])

        # Convert to 2d array
        y_ = y.reshape((-1, 1))

        self.n_features_ = X.shape[1]

        # Get random seed
        rs_ = check_random_state(self.random_state)
        seed_ = rs_.randint(0, np.iinfo('i').max)

        # Define type of data
        fptype = getFPType(X)

        # Fit the model
        train_algo = d4p.gbt_regression_training(fptype=fptype,
                                                 splitMethod=self.splitMethod,
                                                 maxIterations=self.maxIterations,
                                                 maxTreeDepth=self.maxTreeDepth,
                                                 shrinkage=self.shrinkage,
                                                 minSplitLoss=self.minSplitLoss,
                                                 lambda_=self.lambda_,
                                                 observationsPerTreeFraction=self.observationsPerTreeFraction,
                                                 featuresPerNode=self.featuresPerNode,
                                                 minObservationsInLeafNode=self.minObservationsInLeafNode,
                                                 memorySavingMode=self.memorySavingMode,
                                                 maxBins=self.maxBins,
                                                 minBinSize=self.minBinSize,
                                                 engine=d4p.engines_mcg59(seed=seed_))
        train_result = train_algo.compute(X, y_)

        # Store the model
        self.daal_model_ = train_result.model

        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['n_features_'])

        # Input validation
        X = check_array(X, dtype=[np.single, np.double])
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen in `fit`')

        if not hasattr(self, 'daal_model_'):
            raise ValueError(("The class {} instance does not have 'daal_model_' attribute set. "
                              "Call 'fit' with appropriate arguments before using this method.").format(type(self).__name__))

        # Define type of data
        fptype = getFPType(X)

        # Prediction
        predict_algo = d4p.gbt_regression_prediction(fptype=fptype)
        predict_result = predict_algo.compute(X, self.daal_model_)

        return predict_result.prediction.ravel()
