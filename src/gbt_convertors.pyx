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

import json
import re
from collections import deque
from os import getpid, remove
from time import time
from typing import Any, Deque, Dict, List


def get_lightgbm_params(booster):
    return booster.dump_model()

def get_xgboost_params(booster):
    return json.loads(booster.save_config())

def get_catboost_params(booster):
    dump_filename = f"catboost_model_{getpid()}_{time()}"

    # Dump model in file
    booster.save_model(dump_filename, 'json')

    # Read json with model
    with open(dump_filename) as file:
        model_data = json.load(file)

    # Delete dump file
    remove(dump_filename)
    return model_data

def get_gbt_model_from_lightgbm(model: Any, lgb_model = None) -> Any:
    class Node:
        def __init__(self, tree: Dict[str, Any], parent_id: int, position: int):
            self.tree = tree
            self.parent_id = parent_id
            self.position = position

    if lgb_model is None:
        lgb_model = get_lightgbm_params(model)

    n_features = lgb_model["max_feature_idx"] + 1
    n_iterations = len(lgb_model["tree_info"]) / lgb_model["num_tree_per_iteration"]
    n_classes = lgb_model["num_tree_per_iteration"]

    is_regression = False
    objective_fun = lgb_model["objective"]
    if n_classes > 2:
        if "multiclass" not in objective_fun:
            raise TypeError(
                "multiclass (softmax) objective is only supported for multiclass classification")
    elif "binary" in objective_fun:  # nClasses == 1
        n_classes = 2
    else:
        is_regression = True

    if is_regression:
        mb = gbt_reg_model_builder(n_features=n_features, n_iterations=n_iterations)
    else:
        mb = gbt_clf_model_builder(
            n_features=n_features, n_iterations=n_iterations, n_classes=n_classes)

    class_label = 0
    iterations_counter = 0
    for tree in lgb_model["tree_info"]:
        if is_regression:
            tree_id = mb.create_tree(tree["num_leaves"]*2-1)
        else:
            tree_id = mb.create_tree(n_nodes=tree["num_leaves"]*2-1, class_label=class_label)

        iterations_counter += 1
        if iterations_counter == n_iterations:
            iterations_counter = 0
            class_label += 1
        sub_tree = tree["tree_structure"]

        # root is leaf
        if "leaf_value" in sub_tree:
            mb.add_leaf(tree_id=tree_id, response=sub_tree["leaf_value"])
            continue

        # add root
        feat_val = sub_tree["threshold"]
        if isinstance(feat_val, str):
            raise NotImplementedError(
                "Categorical features are not supported in daal4py Gradient Boosting Trees")
        default_left = int(sub_tree["default_left"])
        parent_id = mb.add_split(
            tree_id=tree_id, feature_index=sub_tree["split_feature"],
            feature_value=feat_val, default_left=default_left)

        # create stack
        node_stack: List[Node] = [Node(sub_tree["left_child"], parent_id, 0),
                                  Node(sub_tree["right_child"], parent_id, 1)]

        # dfs through it
        while node_stack:
            sub_tree = node_stack[-1].tree
            parent_id = node_stack[-1].parent_id
            position = node_stack[-1].position
            node_stack.pop()

            # current node is leaf
            if "leaf_index" in sub_tree:
                mb.add_leaf(
                    tree_id=tree_id, response=sub_tree["leaf_value"],
                    parent_id=parent_id, position=position)
                continue

            # current node is split
            feat_val = sub_tree["threshold"]
            if isinstance(feat_val, str):
                raise NotImplementedError(
                    "Categorical features are not supported in daal4py Gradient Boosting Trees")
            default_left = int(sub_tree["default_left"])
            parent_id = mb.add_split(
                tree_id=tree_id, feature_index=sub_tree["split_feature"],
                feature_value=feat_val,
                default_left=default_left,
                parent_id=parent_id, position=position)

            # append children
            node_stack.append(Node(sub_tree["left_child"], parent_id, 0))
            node_stack.append(Node(sub_tree["right_child"], parent_id, 1))

    return mb.model()


def get_gbt_model_from_xgboost(booster: Any, xgb_config=None) -> Any:
    class Node:
        def __init__(self, tree: Dict, parent_id: int, position: int):
            self.tree = tree
            self.parent_id = parent_id
            self.position = position

    # Release Note for XGBoost 1.5.0: Python interface now supports configuring 
    # constraints using feature names instead of feature indices. This also
    # helps with pandas input with set feature names.
    lst = [*range(booster.num_features())]
    booster.feature_names = [str(i) for i in lst]

    trees_arr = booster.get_dump(dump_format="json")
    if xgb_config is None:
        xgb_config = get_xgboost_params(booster)


    n_features = int(xgb_config["learner"]["learner_model_param"]["num_feature"])
    n_classes = int(xgb_config["learner"]["learner_model_param"]["num_class"])
    base_score = float(xgb_config["learner"]["learner_model_param"]["base_score"])

    is_regression = False
    objective_fun = xgb_config["learner"]["learner_train_param"]["objective"]
    if n_classes > 2:
        if objective_fun not in ["multi:softprob", "multi:softmax"]:
            raise TypeError(
                "multi:softprob and multi:softmax are only supported for multiclass classification")
    elif objective_fun.find("binary:") == 0:
        if objective_fun in ["binary:logistic", "binary:logitraw"]:
            n_classes = 2
        else:
            raise TypeError(
                "binary:logistic and binary:logitraw are only supported for binary classification")
    else:
        is_regression = True

    n_iterations = booster.best_iteration + 1
    trees_arr = trees_arr[: n_iterations * (n_classes if n_classes > 2 else 1)]

    # Create + base iteration
    if is_regression:
        mb = gbt_reg_model_builder(n_features=n_features, n_iterations=n_iterations + 1)

        tree_id = mb.create_tree(1)
        mb.add_leaf(tree_id=tree_id, response=base_score)
    else:
        mb = gbt_clf_model_builder(
            n_features=n_features, n_iterations=n_iterations, n_classes=n_classes)

    class_label = 0
    iterations_counter = 0
    mis_eq_yes = None
    for tree in trees_arr:
        n_nodes = 1
        # find out the number of nodes in the tree
        for node in tree.split("nodeid")[1:]:
            node_id = int(node[3:node.find(",")])
            if node_id + 1 > n_nodes:
                n_nodes = node_id + 1
        if is_regression:
            tree_id = mb.create_tree(n_nodes)
        else:
            tree_id = mb.create_tree(n_nodes=n_nodes, class_label=class_label)

        iterations_counter += 1
        if iterations_counter == n_iterations:
            iterations_counter = 0
            class_label += 1
        sub_tree = json.loads(tree)

        # root is leaf
        if "leaf" in sub_tree:
            mb.add_leaf(tree_id=tree_id, response=sub_tree["leaf"])
            continue

        # add root
        try:
            feature_index = int(sub_tree["split"])
        except ValueError:
            raise TypeError("Feature names must be integers")
        feature_value = np.nextafter(np.single(sub_tree["split_condition"]), np.single(-np.inf))
        default_left = int(sub_tree["yes"] == sub_tree["missing"])
        parent_id = mb.add_split(tree_id=tree_id, feature_index=feature_index,
                                 feature_value=feature_value, default_left=default_left)

        # create queue
        node_queue: Deque[Node] = deque()
        node_queue.append(Node(sub_tree["children"][0], parent_id, 0))
        node_queue.append(Node(sub_tree["children"][1], parent_id, 1))

        # bfs through it
        while node_queue:
            sub_tree = node_queue[0].tree
            parent_id = node_queue[0].parent_id
            position = node_queue[0].position
            node_queue.popleft()

            # current node is leaf
            if "leaf" in sub_tree:
                mb.add_leaf(
                    tree_id=tree_id, response=sub_tree["leaf"],
                    parent_id=parent_id, position=position)
                continue

            # current node is split
            try:
                feature_index = int(sub_tree["split"])
            except ValueError:
                raise TypeError("Feature names must be integers")
            feature_value = np.nextafter(np.single(sub_tree["split_condition"]), np.single(-np.inf))
            default_left = int(sub_tree["yes"] == sub_tree["missing"])

            parent_id = mb.add_split(
                tree_id=tree_id, feature_index=feature_index, feature_value=feature_value,
                default_left=default_left, parent_id=parent_id, position=position)

            # append to queue
            node_queue.append(Node(sub_tree["children"][0], parent_id, 0))
            node_queue.append(Node(sub_tree["children"][1], parent_id, 1))

    return mb.model()

def get_gbt_model_from_catboost(model: Any, model_data=None) -> Any:
    if not model.is_fitted():
        raise RuntimeError(
            "Model should be fitted before exporting to daal4py.")

    if model_data is None:
        model_data = get_catboost_params(model)

    if 'categorical_features' in model_data['features_info']:
        raise NotImplementedError(
            "Categorical features are not supported in daal4py Gradient Boosting Trees")

    n_features = len(model_data['features_info']['float_features'])

    is_symmetric_tree = model_data['model_info']['params']['tree_learner_options']['grow_policy'] == 'SymmetricTree'

    if is_symmetric_tree:
        n_iterations = len(model_data['oblivious_trees'])
    else:
        n_iterations = len(model_data['trees'])

    n_classes = 0

    if 'class_params' in model_data['model_info']:
        is_classification = True
        n_classes = len(model_data['model_info']
                        ['class_params']['class_to_label'])
        mb = gbt_clf_model_builder(
            n_features=n_features, n_iterations=n_iterations, n_classes=n_classes)
    else:
        is_classification = False
        mb = gbt_reg_model_builder(n_features, n_iterations)

    splits = []

    # Create splits array (all splits are placed sequentially)
    for feature in model_data['features_info']['float_features']:
        if feature['borders']:
            for feature_border in feature['borders']:
                splits.append(
                    {'feature_index': feature['feature_index'], 'value': feature_border})

    if not is_classification:
        bias = model_data['scale_and_bias'][1][0] / n_iterations
        scale = model_data['scale_and_bias'][0]
    else:
        bias = 0
        scale = 1

    trees_explicit = []
    tree_symmetric = []

    if model_data['model_info']['params']['data_processing_options']['float_features_binarization']['nan_mode'] == 'Min':
        default_left = 1
    else:
        default_left = 0

    for tree_num in range(n_iterations):
        if is_symmetric_tree:
            
            if model_data['oblivious_trees'][tree_num]['splits'] is not None:
                # Tree has more than 1 node
                cur_tree_depth = len(
                    model_data['oblivious_trees'][tree_num]['splits'])
            else:
                cur_tree_depth = 0

            tree_symmetric.append(
                (model_data['oblivious_trees'][tree_num], cur_tree_depth))
        else:
            class Node:
                def __init__(self, parent=None, split=None, value=None) -> None:
                    self.right = None
                    self.left = None
                    self.split = split
                    self.value = value

            n_nodes = 1
            # Check if node is a leaf (in case of stump)
            if 'split' in model_data['trees'][tree_num]:
                # Get number of trees and splits info via BFS
                # Create queue
                nodes_queue = []
                root_node = Node(
                    split=splits[model_data['trees'][tree_num]['split']['split_index']])
                nodes_queue.append((model_data['trees'][tree_num], root_node))
                while nodes_queue:
                    cur_node_data, cur_node = nodes_queue.pop(0)
                    if 'value' in cur_node_data:
                        if isinstance(cur_node_data['value'], list):
                            cur_node.value = [
                                value for value in cur_node_data['value']]
                        else:
                            cur_node.value = [
                                cur_node_data['value'] * scale + bias]
                    else:
                        cur_node.split = splits[cur_node_data['split']
                                                ['split_index']]
                        left_node = Node()
                        right_node = Node()
                        cur_node.left = left_node
                        cur_node.right = right_node
                        nodes_queue.append((cur_node_data['left'], left_node))
                        nodes_queue.append(
                            (cur_node_data['right'], right_node))
                        n_nodes += 2
            else:
                root_node = Node()
                if is_classification and n_classes > 2:
                    root_node.value = [
                        value * scale for value in model_data['trees'][tree_num]['value']]
                else:
                    root_node.value = [model_data['trees'][tree_num]['value'] * scale + bias]
            trees_explicit.append((root_node, n_nodes))

    tree_id = []
    class_label = 0
    count = 0

    # Only 1 tree for each iteration in case of regression or binary classification
    if not is_classification or n_classes == 2:
        n_tree_each_iter = 1
    else:
        n_tree_each_iter = n_classes

    # Create id for trees (for the right order in modelbuilder)
    for i in range(n_iterations):
        for c in range(n_tree_each_iter):
            if is_symmetric_tree:
                n_nodes = 2**(tree_symmetric[i][1] + 1) - 1
            else:
                n_nodes = trees_explicit[i][1]

            if is_classification and n_classes > 2:
                tree_id.append(mb.create_tree(n_nodes, class_label))
                count += 1
                if count == n_iterations:
                    class_label += 1
                    count = 0

            elif is_classification:
                tree_id.append(mb.create_tree(n_nodes, 0))
            else:
                tree_id.append(mb.create_tree(n_nodes))
    

    if is_symmetric_tree:
        for class_label in range(n_tree_each_iter):
            for i in range(n_iterations):
                cur_tree_info = tree_symmetric[i][0]
                cur_tree_id = tree_id[i * n_tree_each_iter + class_label]
                cur_tree_leaf_val = cur_tree_info['leaf_values']
                cur_tree_depth = tree_symmetric[i][1]

                if cur_tree_depth == 0:
                    mb.add_leaf(
                        tree_id=cur_tree_id, response=cur_tree_leaf_val[0])
                else:
                    # One split used for the whole level 
                    cur_level_split = splits[cur_tree_info['splits']
                                             [cur_tree_depth - 1]['split_index']]
                    root_id = mb.add_split(
                        tree_id=cur_tree_id, feature_index=cur_level_split['feature_index'], feature_value=cur_level_split['value'],
                        default_left=default_left)
                    prev_level_nodes = [root_id]

                    # Iterate over levels, splits in json are reversed (root split is the last)
                    for cur_level in range(cur_tree_depth - 2, -1, -1):
                        cur_level_nodes = []
                        for cur_parent in prev_level_nodes:
                            cur_level_split = splits[cur_tree_info['splits']
                                                     [cur_level]['split_index']]
                            cur_left_node = mb.add_split(tree_id=cur_tree_id, parent_id=cur_parent, position=0,
                                                         feature_index=cur_level_split['feature_index'], feature_value=cur_level_split['value'],
                                                         default_left=default_left)
                            cur_right_node = mb.add_split(tree_id=cur_tree_id, parent_id=cur_parent, position=1,
                                                          feature_index=cur_level_split['feature_index'], feature_value=cur_level_split['value'],
                                                          default_left=default_left)
                            cur_level_nodes.append(cur_left_node)
                            cur_level_nodes.append(cur_right_node)
                        prev_level_nodes = cur_level_nodes

                    # Different storing format for leaves
                    if not is_classification or n_classes == 2:
                        for last_level_node_num in range(len(prev_level_nodes)):
                            mb.add_leaf(tree_id=cur_tree_id, response=cur_tree_leaf_val[2 * last_level_node_num]
                                        * scale + bias, parent_id=prev_level_nodes[last_level_node_num], position=0)
                            mb.add_leaf(tree_id=cur_tree_id, response=cur_tree_leaf_val[2 * last_level_node_num + 1]
                                        * scale + bias, parent_id=prev_level_nodes[last_level_node_num], position=1)
                    else:
                        for last_level_node_num in range(len(prev_level_nodes)):
                            left_index = 2 * last_level_node_num * n_tree_each_iter + class_label
                            right_index = (2 * last_level_node_num + 1) * \
                                n_tree_each_iter + class_label
                            mb.add_leaf(
                                tree_id=cur_tree_id, response=cur_tree_leaf_val[left_index] * scale + bias, parent_id=prev_level_nodes[last_level_node_num], position=0)
                            mb.add_leaf(
                                tree_id=cur_tree_id, response=cur_tree_leaf_val[right_index] * scale + bias, parent_id=prev_level_nodes[last_level_node_num], position=1)
    else:
        for class_label in range(n_tree_each_iter):
            for i in range(n_iterations):
                root_node = trees_explicit[i][0]

                cur_tree_id = tree_id[i * n_tree_each_iter + class_label]
                # Traverse tree via BFS and build tree with modelbuilder
                if root_node.value is None:
                    root_id = mb.add_split(
                        tree_id=cur_tree_id, feature_index=root_node.split['feature_index'], feature_value=root_node.split['value'],
                        default_left=default_left)
                    nodes_queue = [(root_node, root_id)]
                    while nodes_queue:
                        cur_node, cur_node_id = nodes_queue.pop(0)
                        left_node = cur_node.left
                        # Check if node is a leaf
                        if left_node.value is None:
                            left_node_id = mb.add_split(tree_id=cur_tree_id, parent_id=cur_node_id, position=0,
                                                        feature_index=left_node.split['feature_index'], feature_value=left_node.split['value'],
                                                        default_left=default_left)
                            nodes_queue.append((left_node, left_node_id))
                        else:
                            mb.add_leaf(
                                tree_id=cur_tree_id, response=left_node.value[class_label], parent_id=cur_node_id, position=0)
                        right_node = cur_node.right
                        # Check if node is a leaf
                        if right_node.value is None:
                            right_node_id = mb.add_split(tree_id=cur_tree_id, parent_id=cur_node_id, position=1,
                                                         feature_index=right_node.split['feature_index'], feature_value=right_node.split['value'],
                                                         default_left=default_left)
                            nodes_queue.append((right_node, right_node_id))
                        else:
                            mb.add_leaf(
                                tree_id=cur_tree_id, response=cur_node.right.value[class_label],
                                parent_id=cur_node_id, position=1)

                else:
                    # Tree has only one node
                    mb.add_leaf(tree_id=cur_tree_id,
                                response=root_node.value[class_label])

    return mb.model()
