#===============================================================================
# Copyright 2020-2021 Intel Corporation
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

from typing import List, Deque, Dict, Any
from collections import deque
from os import remove
import json
import re

def get_gbt_model_from_lightgbm(model: Any) -> Any:
    class Node:
        def __init__(self, tree: Dict[str, Any], parent_id: int, position: int):
            self.tree = tree
            self.parent_id = parent_id
            self.position = position

    lgb_model = model.dump_model()

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
        tree_struct = tree["tree_structure"]

        # root is leaf
        if "leaf_value" in tree_struct:
            mb.add_leaf(tree_id=tree_id, response=tree_struct["leaf_value"])
            continue

        # add root
        feat_val = tree_struct["threshold"]
        if isinstance(feat_val, str):
            raise NotImplementedError(
                "Categorical features are not supported in daal4py Gradient Boosting Trees")
        parent_id = mb.add_split(
            tree_id=tree_id, feature_index=tree_struct["split_feature"],
            feature_value=feat_val)

        # create stack
        node_stack: List[Node] = [Node(tree_struct["left_child"], parent_id, 0),
                                  Node(tree_struct["right_child"], parent_id, 1)]

        # dfs through it
        while node_stack:
            tree_struct = node_stack[-1].tree
            parent_id = node_stack[-1].parent_id
            position = node_stack[-1].position
            node_stack.pop()

            # current node is leaf
            if "leaf_index" in tree_struct:
                mb.add_leaf(
                    tree_id=tree_id, response=tree_struct["leaf_value"],
                    parent_id=parent_id, position=position)
                continue

            # current node is split
            feat_val = tree_struct["threshold"]
            if isinstance(feat_val, str):
                raise NotImplementedError(
                    "Categorical features are not supported in daal4py Gradient Boosting Trees")
            parent_id = mb.add_split(
                tree_id=tree_id, feature_index=tree_struct["split_feature"],
                feature_value=feat_val,
                parent_id=parent_id, position=position)

            # append children
            node_stack.append(Node(tree_struct["left_child"], parent_id, 0))
            node_stack.append(Node(tree_struct["right_child"], parent_id, 1))

    return mb.model()


def get_gbt_model_from_xgboost(booster: Any) -> Any:
    class Node:
        def __init__(self, tree: str, parent_id: int, position: int):
            self.tree = tree
            self.parent_id = parent_id
            self.position = position

    booster.dump_model("raw.txt")
    with open("raw.txt", "r") as read_file:
        xgb_model = read_file.read()
    remove("./raw.txt")
    xgb_config = json.loads(booster.save_config())

    updater = list(xgb_config["learner"]["gradient_booster"]["updater"].keys())[0]
    max_depth = int(xgb_config["learner"]["gradient_booster"]
                    ["updater"][updater]["train_param"]["max_depth"])
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

    last_booster_start = xgb_model.rfind("booster[") + 8
    n_iterations = int((int(xgb_model[last_booster_start:xgb_model.find(
        "]", last_booster_start)]) + 1) / (n_classes if n_classes > 2 else 1))

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
    for tree in xgb_model.expandtabs(1).replace(" ", "").split("booster[")[1:]:
        if is_regression:
            tree_id = mb.create_tree(len(tree.split("\n")) - 1)
        else:
            tree_id = mb.create_tree(n_nodes=len(tree.split("\n")) - 1, class_label=class_label)

        iterations_counter += 1
        if iterations_counter == n_iterations:
            iterations_counter = 0
            class_label += 1
        sub_tree = tree[tree.find("\n")+1:]

        # root is leaf
        if sub_tree.find("leaf") == sub_tree.find(":") + 1:
            mb.add_leaf(
                tree_id=tree_id, response=float(
                    sub_tree[sub_tree.find("=") + 1: sub_tree.find("\n")]))
            continue

        # add root
        try:
            feature_name = sub_tree[sub_tree.find("[") + 1:sub_tree.find("<")]
            str_index = re.sub(r'[^0-9]', '', feature_name)
            feature_index = int(str_index)
        except ValueError:
            raise TypeError("Feature names must be integers")
        feature_value = np.nextafter(
            np.single(sub_tree[sub_tree.find("<") + 1: sub_tree.find("]")]),
            np.single(-np.inf))
        parent_id = mb.add_split(tree_id=tree_id, feature_index=feature_index,
                                 feature_value=feature_value)

        # create queue
        yes_idx = sub_tree[sub_tree.find("yes=") + 4:sub_tree.find(",no")]
        no_idx = sub_tree[sub_tree.find("no=") + 3:sub_tree.find(",missing")]
        mis_idx = sub_tree[sub_tree.find("missing=") + 8:sub_tree.find("\n")]
        if mis_eq_yes is None:
            if mis_idx == yes_idx:
                mis_eq_yes = True
            elif mis_idx == no_idx:
                mis_eq_yes = False
            else:
                raise TypeError("Missing values are not supported in daal4py Gradient Boosting Trees")
        elif mis_eq_yes and mis_idx != yes_idx or not mis_eq_yes and mis_idx != no_idx:
            raise TypeError("Missing values are not supported in daal4py Gradient Boosting Trees")
        node_queue: Deque[Node] = deque()
        node_queue.append(
            Node(
                sub_tree
                [sub_tree.find("\n" + yes_idx + ":") + 1: sub_tree.find("\n" + no_idx + ":") + 1],
                parent_id, 0))
        node_queue.append(Node(sub_tree[sub_tree.find("\n" + no_idx + ":") + 1:], parent_id, 1))

        # bfs through it
        while node_queue:
            sub_tree = node_queue[0].tree
            parent_id = node_queue[0].parent_id
            position = node_queue[0].position
            node_queue.popleft()

            # current node is leaf
            if sub_tree.find("leaf") == sub_tree.find(":") + 1:
                mb.add_leaf(
                    tree_id=tree_id,
                    response=float(sub_tree[sub_tree.find("=") + 1: sub_tree.find("\n")]),
                    parent_id=parent_id, position=position)
                continue

            # current node is split
            try:
                feature_name = sub_tree[sub_tree.find("[") + 1:sub_tree.find("<")]
                str_index = re.sub(r'[^0-9]', '', feature_name)
                feature_index = int(str_index)
            except ValueError:
                raise TypeError("Feature names must be integers")
            feature_value = np.nextafter(
                np.single(sub_tree[sub_tree.find("<") + 1: sub_tree.find("]")]),
                np.single(-np.inf))
            parent_id = mb.add_split(
                tree_id=tree_id, feature_index=feature_index, feature_value=feature_value,
                parent_id=parent_id, position=position)

            # append to queue
            yes_idx = sub_tree[sub_tree.find("yes=") + 4:sub_tree.find(",no")]
            no_idx = sub_tree[sub_tree.find("no=") + 3:sub_tree.find(",missing")]
            mis_idx = sub_tree[sub_tree.find("missing=") + 8:sub_tree.find("\n")]
            if mis_eq_yes and mis_idx != yes_idx or not mis_eq_yes and mis_idx != no_idx:
                raise TypeError("Missing values are not supported in daal4py Gradient Boosting Trees")
            node_queue.append(
                Node(
                    sub_tree
                    [sub_tree.find("\n" + yes_idx + ":") + 1: sub_tree.find("\n" + no_idx + ":") + 1],
                    parent_id, 0))
            node_queue.append(
                Node(
                    sub_tree[sub_tree.find("\n" + no_idx + ":") + 1:],
                    parent_id, 1))

    return mb.model()
