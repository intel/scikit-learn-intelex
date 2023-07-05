#===============================================================================
# Copyright 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

# We expose DAAL's directly through Cython
# The model builder object is retrieved through calling model_builder.
# We will extend this once we know how other model builders will work in DAAL

import json

cdef extern from "gbt_model_builder.h":
    ctypedef size_t c_gbt_clf_node_id
    ctypedef size_t c_gbt_clf_tree_id
    ctypedef size_t c_gbt_reg_node_id
    ctypedef size_t c_gbt_reg_tree_id

    cdef size_t c_gbt_clf_no_parent
    cdef size_t c_gbt_reg_no_parent

    cdef gbt_classification_ModelPtr * get_gbt_classification_model_builder_model(c_gbt_classification_model_builder *)
    cdef gbt_regression_ModelPtr * get_gbt_regression_model_builder_model(c_gbt_regression_model_builder *)

    cdef cppclass c_gbt_classification_model_builder:
        c_gbt_classification_model_builder(size_t nFeatures, size_t nIterations, size_t nClasses) except +
        c_gbt_clf_tree_id createTree(size_t nNodes, size_t classLabel)
        c_gbt_clf_node_id addLeafNode(c_gbt_clf_tree_id treeId, c_gbt_clf_node_id parentId, size_t position, double response)

    cdef cppclass c_gbt_regression_model_builder:
        c_gbt_regression_model_builder(size_t nFeatures, size_t nIterations) except +
        c_gbt_reg_tree_id createTree(size_t nNodes)
        c_gbt_reg_node_id addLeafNode(c_gbt_reg_tree_id treeId, c_gbt_reg_node_id parentId, size_t position, double response)

    cdef c_gbt_clf_node_id clfAddSplitNodeWrapper(c_gbt_classification_model_builder * c_ptr, c_gbt_clf_tree_id treeId, c_gbt_clf_node_id parentId, size_t position, size_t featureIndex, double featureValue, int defaultLeft)
    cdef c_gbt_reg_node_id regAddSplitNodeWrapper(c_gbt_regression_model_builder     * c_ptr, c_gbt_reg_tree_id treeId, c_gbt_reg_node_id parentId, size_t position, size_t featureIndex, double featureValue, int defaultLeft)

cdef class gbt_classification_model_builder:
    '''
    Model Builder for gradient boosted trees.
    '''
    cdef c_gbt_classification_model_builder * c_ptr

    def __cinit__(self, size_t n_features, size_t n_iterations, size_t n_classes):
        self.c_ptr = new c_gbt_classification_model_builder(n_features, n_iterations, n_classes)

    def __dealloc__(self):
        del self.c_ptr

    def create_tree(self, size_t n_nodes, size_t class_label):
        '''
        Create certain tree in the gradient boosted trees classification model for certain class

        :param size_t n_nodes: number of nodes in created tree
        :param size_t class_label: label of class for which tree is created. classLabel bellows interval from 0 to (nClasses - 1)
        :rtype: tree identifier
        '''
        return self.c_ptr.createTree(n_nodes, class_label)

    def add_leaf(self, c_gbt_clf_tree_id tree_id, double response, c_gbt_clf_node_id parent_id=c_gbt_clf_no_parent, size_t position=0):
        '''
        Create Leaf node and add it to certain tree

        :param tree-handle tree_id: tree to which new node is added
        :param node-handle parent_id: parent node to which new node is added (use noParent for root node)
        :param size_t position: position in parent (e.g. 0 for left and 1 for right child in a binary tree)
        :param double response: response value for leaf node to be predicted
        :rtype: node identifier
        '''
        return self.c_ptr.addLeafNode(tree_id, parent_id, position, response)

    def add_split(self, c_gbt_clf_tree_id tree_id, size_t feature_index, double feature_value, int default_left, c_gbt_clf_node_id parent_id=c_gbt_clf_no_parent, size_t position=0):
        '''
        Create Split node and add it to certain tree.

        :param tree-handle tree_id: tree to which node is added
        :param node-handle parent_id: parent node to which new node is added (use noParent for root node)
        :param size_t position: position in parent (e.g. 0 for left and 1 for right child in a binary tree)
        :param size_t feature_index: feature index for spliting
        :param double feature_value: feature value for spliting
        :param int default_left: default behaviour in case of missing value
        :rtype: node identifier
        '''
        return clfAddSplitNodeWrapper(self.c_ptr, tree_id, parent_id, position, feature_index, feature_value, default_left)

    def model(self):
        '''
        Get built model

        :rtype: gbt_classification_model
        '''
        cdef gbt_classification_model res = gbt_classification_model.__new__(gbt_classification_model)
        res.c_ptr = get_gbt_classification_model_builder_model(self.c_ptr)
        return res


cdef class gbt_regression_model_builder:
    '''
    Model Builder for gradient boosted trees.
    '''
    cdef c_gbt_regression_model_builder * c_ptr

    def __cinit__(self, size_t n_features, size_t n_iterations):
        self.c_ptr = new c_gbt_regression_model_builder(n_features, n_iterations)

    def __dealloc__(self):
        del self.c_ptr

    def create_tree(self, size_t n_nodes):
        '''
        Create certain tree in the gradient boosted trees regression model

        :param size_t n_nodes: number of nodes in created tree
        :rtype: tree identifier
        '''
        return self.c_ptr.createTree(n_nodes)

    def add_leaf(self, c_gbt_reg_tree_id tree_id, double response, c_gbt_reg_node_id parent_id=c_gbt_reg_no_parent, size_t position=0):
        '''
        Create Leaf node and add it to certain tree

        :param tree-handle tree_id: tree to which new node is added
        :param node-handle parent_id: parent node to which new node is added (use noParent for root node)
        :param size_t position: position in parent (e.g. 0 for left and 1 for right child in a binary tree)
        :param double response: response value for leaf node to be predicted
        :rtype: node identifier
        '''
        return self.c_ptr.addLeafNode(tree_id, parent_id, position, response)

    def add_split(self, c_gbt_reg_tree_id tree_id, size_t feature_index, double feature_value, int default_left, c_gbt_reg_node_id parent_id=c_gbt_reg_no_parent, size_t position=0):
        '''
        Create Split node and add it to certain tree.

        :param tree-handle tree_id: tree to which node is added
        :param node-handle parent_id: parent node to which new node is added (use noParent for root node)
        :param size_t position: position in parent (e.g. 0 for left and 1 for right child in a binary tree)
        :param size_t feature_index: feature index for spliting
        :param double feature_value: feature value for spliting
        :param int default_left: default behaviour in case of missing value
        :rtype: node identifier
        '''
        return regAddSplitNodeWrapper(self.c_ptr, tree_id, parent_id, position, feature_index, feature_value, default_left)

    def model(self):
        '''
        Get built model

        :rtype: gbt_regression_model
        '''
        cdef gbt_regression_model res = gbt_regression_model.__new__(gbt_regression_model)
        res.c_ptr = get_gbt_regression_model_builder_model(self.c_ptr)
        return res


def gbt_clf_model_builder(n_features, n_iterations, n_classes = 2):
    '''
    Model builder for gradient boosted trees classification

    :param size_t n_features: Number of features in training data
    :param size_t n_iterations: Number of trees in model for each class
    :param size_t n_classes: Number of classes in model
    '''
    return gbt_classification_model_builder(n_features, n_iterations, n_classes)


def gbt_reg_model_builder(n_features, n_iterations):
    '''
    Model builder for gradient boosted trees regression

    :param size_t n_features: Number of features in training data
    :param size_t n_iterations: Number of trees in the model
    '''
    return gbt_regression_model_builder(n_features, n_iterations)


class GBTDAALBaseModel:
    def _get_params_from_lightgbm(self, params):
        self.n_classes_ = params["num_tree_per_iteration"]
        objective_fun = params["objective"]
        if self.n_classes_ <= 2:
            if "binary" in objective_fun:  # nClasses == 1
                self.n_classes_ = 2

        self.n_features_in_ = params["max_feature_idx"] + 1

    def _get_params_from_xgboost(self, params):
        self.n_classes_ = int(params["learner"]["learner_model_param"]["num_class"])
        objective_fun = params["learner"]["learner_train_param"]["objective"]
        if self.n_classes_ <= 2:
            if objective_fun in ["binary:logistic", "binary:logitraw"]:
                self.n_classes_ = 2

        self.n_features_in_ = int(params["learner"]["learner_model_param"]["num_feature"])

    def _get_params_from_catboost(self, params):
        if 'class_params' in params['model_info']:
            self.n_classes_ = len(params['model_info']['class_params']['class_to_label'])
        self.n_features_in_ = len(params['features_info']['float_features'])

    def _convert_model_from_lightgbm(self, booster):
        lgbm_params = get_lightgbm_params(booster)
        self.daal_model_ = get_gbt_model_from_lightgbm(booster, lgbm_params)
        self._get_params_from_lightgbm(lgbm_params)

    def _convert_model_from_xgboost(self, booster):
        xgb_params = get_xgboost_params(booster)
        self.daal_model_ = get_gbt_model_from_xgboost(booster, xgb_params)
        self._get_params_from_xgboost(xgb_params)

    def _convert_model_from_catboost(self, booster):
        catboost_params = get_catboost_params(booster)
        self.daal_model_ = get_gbt_model_from_catboost(booster)
        self._get_params_from_catboost(catboost_params)

    def _convert_model(self, model):
        (submodule_name, class_name) = (model.__class__.__module__,
                                        model.__class__.__name__)
        self_class_name = self.__class__.__name__

        # Build GBTDAALClassifier from LightGBM
        if (submodule_name, class_name) == ("lightgbm.sklearn", "LGBMClassifier"):
            if self_class_name == "GBTDAALClassifier":
                self._convert_model_from_lightgbm(model.booster_)
            else:
                raise TypeError(f"Only GBTDAALClassifier can be created from {submodule_name}.{class_name} (got {self_class_name})")
        # Build GBTDAALClassifier from XGBoost
        elif (submodule_name, class_name) == ("xgboost.sklearn", "XGBClassifier"):
            if self_class_name == "GBTDAALClassifier":
                self._convert_model_from_xgboost(model.get_booster())
            else:
                raise TypeError(f"Only GBTDAALClassifier can be created from {submodule_name}.{class_name} (got {self_class_name})")
        # Build GBTDAALClassifier from CatBoost
        elif (submodule_name, class_name) == ("catboost.core", "CatBoostClassifier"):
            if self_class_name == "GBTDAALClassifier":
                self._convert_model_from_catboost(model)
            else:
                raise TypeError(f"Only GBTDAALClassifier can be created from {submodule_name}.{class_name} (got {self_class_name})")
        # Build GBTDAALRegressor from LightGBM
        elif (submodule_name, class_name) == ("lightgbm.sklearn", "LGBMRegressor"):
            if self_class_name == "GBTDAALRegressor":
                self._convert_model_from_lightgbm(model.booster_)
            else:
                raise TypeError(f"Only GBTDAALRegressor can be created from {submodule_name}.{class_name} (got {self_class_name})")
        # Build GBTDAALRegressor from XGBoost
        elif (submodule_name, class_name) == ("xgboost.sklearn", "XGBRegressor"):
            if self_class_name == "GBTDAALRegressor":
                self._convert_model_from_xgboost(model.get_booster())
            else:
                raise TypeError(f"Only GBTDAALRegressor can be created from {submodule_name}.{class_name} (got {self_class_name})")
        # Build GBTDAALRegressor from CatBoost
        elif (submodule_name, class_name) == ("catboost.core", "CatBoostRegressor"):
            if self_class_name == "GBTDAALRegressor":
                self._convert_model_from_catboost(model)
            else:
                raise TypeError(f"Only GBTDAALRegressor can be created from {submodule_name}.{class_name} (got {self_class_name})")
        # Build GBTDAALModel from LightGBM
        elif (submodule_name, class_name) == ("lightgbm.basic", "Booster"):
            if self_class_name == "GBTDAALModel":
                self._convert_model_from_lightgbm(model)
            else:
                raise TypeError(f"Only GBTDAALModel can be created from {submodule_name}.{class_name} (got {self_class_name})")
        # Build GBTDAALModel from XGBoost
        elif (submodule_name, class_name) == ("xgboost.core", "Booster"):
            if self_class_name == "GBTDAALModel":
                self._convert_model_from_xgboost(model)
            else:
                raise TypeError(f"Only GBTDAALModel can be created from {submodule_name}.{class_name} (got {self_class_name})")
        # Build GBTDAALModel from CatBoost
        elif (submodule_name, class_name) == ("catboost.core", "CatBoost"):
            if self_class_name == "GBTDAALModel":
                self._convert_model_from_catboost(model)
            else:
                raise TypeError(f"Only GBTDAALModel can be created from {submodule_name}.{class_name} (got {self_class_name})")
        else:
            raise TypeError(f"Unknown model format {submodule_name}.{class_name}")



    def _predict_classification(self, X, fptype, resultsToEvaluate):
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Shape of input is different from what was seen in `fit`')

        if not hasattr(self, 'daal_model_'):
            raise ValueError((
                "The class {} instance does not have 'daal_model_' attribute set. "
                "Call 'fit' with appropriate arguments before using this method.").format(
                    type(self).__name__))

        # Prediction
        predict_algo = gbt_classification_prediction(
            fptype=fptype,
            nClasses=self.n_classes_,
            resultsToEvaluate=resultsToEvaluate)
        predict_result = predict_algo.compute(X, self.daal_model_)

        if resultsToEvaluate == "computeClassLabels":
             return predict_result.prediction.ravel().astype(np.int64, copy=False)
        else:
            return predict_result.probabilities

    def _predict_regression(self, X, fptype):
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Shape of input is different from what was seen in `fit`')

        if not hasattr(self, 'daal_model_'):
            raise ValueError((
                "The class {} instance does not have 'daal_model_' attribute set. "
                "Call 'fit' with appropriate arguments before using this method.").format(
                    type(self).__name__))

        # Prediction
        predict_algo = gbt_regression_prediction(fptype=fptype)
        predict_result = predict_algo.compute(X, self.daal_model_)

        return predict_result.prediction.ravel()


class GBTDAALModel(GBTDAALBaseModel):
    def convert_model(model):
        gbm = GBTDAALModel()
        gbm._convert_model(model)

        gbm._is_regression = isinstance(gbm.daal_model_, gbt_regression_model)

        return gbm

    def predict(self, X, fptype="float"):
        if self._is_regression:
            return self._predict_regression(X, fptype)
        else:
            return self._predict_classification(X, fptype, "computeClassLabels")

    def predict_proba(self, X, fptype="float"):
        if self._is_regression:
            raise NotImplementedError("Can't predict probabilities for regression task")
        else:
            return self._predict_classification(X, fptype, "computeClassProbabilities")
