# ===============================================================================
# Copyright 2023 Intel Corporation
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
# ===============================================================================

import json
from collections import deque
from tempfile import NamedTemporaryFile
from typing import Any, Deque, Dict, List, Optional, Tuple
from warnings import warn

import numpy as np


class CatBoostNode:
    def __init__(
        self,
        split: Optional[Dict] = None,
        value: Optional[List[float]] = None,
        right: Optional[int] = None,
        left: Optional[float] = None,
        cover: Optional[float] = None,
    ) -> None:
        self.split = split
        self.value = value
        self.right = right
        self.left = left
        self.cover = cover


class Node:
    """Helper class holding Tree Node information"""

    def __init__(
        self,
        cover: float,
        is_leaf: bool,
        default_left: bool,
        feature: int,
        value: float,
        n_children: int = 0,
        left_child: "Optional[Node]" = None,
        right_child: "Optional[Node]" = None,
        parent_id: Optional[int] = -1,
        position: Optional[int] = -1,
    ) -> None:
        self.cover = cover
        self.is_leaf = is_leaf
        self.default_left = default_left
        self.__feature = feature
        self.value = value
        self.n_children = n_children
        self.left_child = left_child
        self.right_child = right_child
        self.parent_id = parent_id
        self.position = position

    @staticmethod
    def from_xgb_dict(input_dict: Dict[str, Any]) -> "Node":
        if "children" in input_dict:
            left_child = Node.from_xgb_dict(input_dict["children"][0])
            right_child = Node.from_xgb_dict(input_dict["children"][1])
            n_children = 2 + left_child.n_children + right_child.n_children
        else:
            left_child = None
            right_child = None
            n_children = 0
        is_leaf = "leaf" in input_dict
        default_left = "yes" in input_dict and input_dict["yes"] == input_dict["missing"]
        return Node(
            cover=input_dict["cover"],
            is_leaf=is_leaf,
            default_left=default_left,
            feature=input_dict.get("split"),
            value=input_dict["leaf"] if is_leaf else input_dict["split_condition"],
            n_children=n_children,
            left_child=left_child,
            right_child=right_child,
        )

    @staticmethod
    def from_lightgbm_dict(input_dict: Dict[str, Any]) -> "Node":
        if "tree_structure" in input_dict:
            tree = input_dict["tree_structure"]
        else:
            tree = input_dict

        n_children = 0
        if "left_child" in tree:
            left_child = Node.from_lightgbm_dict(tree["left_child"])
            n_children += 1 + left_child.n_children
        else:
            left_child = None
        if "right_child" in tree:
            right_child = Node.from_lightgbm_dict(tree["right_child"])
            n_children += 1 + right_child.n_children
        else:
            right_child = None

        is_leaf = "leaf_value" in tree
        return Node(
            cover=tree["leaf_count"] if is_leaf else tree["internal_count"],
            is_leaf=is_leaf,
            default_left=tree.get("default_left", 0),
            feature=tree.get("split_feature"),
            value=tree["leaf_value"] if is_leaf else tree["threshold"],
            n_children=n_children,
            left_child=left_child,
            right_child=right_child,
        )

    def get_value_closest_float_downward(self) -> np.float64:
        """Get the closest exact fp value smaller than self.value"""
        return np.nextafter(np.single(self.value), np.single(-np.inf))

    def get_children(self) -> "Optional[Tuple[Node, Node]]":
        if not self.left_child or not self.right_child:
            assert self.is_leaf
        else:
            return (self.left_child, self.right_child)

    @property
    def feature(self) -> int:
        if isinstance(self.__feature, int):
            return self.__feature
        if isinstance(self.__feature, str) and self.__feature.isnumeric():
            return int(self.__feature)
        raise ValueError(
            f"Feature names must be integers (got ({type(self.__feature)}){self.__feature})"
        )


class TreeView:
    """Helper class, treating a list of nodes as one tree"""

    def __init__(self, tree_id: int, root_node: Node) -> None:
        self.tree_id = tree_id
        self.root_node = root_node

    @property
    def is_leaf(self) -> bool:
        return self.root_node.is_leaf

    @property
    def value(self) -> float:
        if not self.is_leaf:
            raise ValueError("Tree is not a leaf-only tree")
        if self.root_node.value is None:
            raise ValueError("Tree is leaf-only but leaf node has no value")
        return self.root_node.value

    @property
    def cover(self) -> float:
        if not self.is_leaf:
            raise ValueError("Tree is not a leaf-only tree")
        return self.root_node.cover

    @property
    def n_nodes(self) -> int:
        return self.root_node.n_children + 1


class TreeList(list):
    """Helper class that is able to extract all information required by the
    model builders from various objects"""

    @staticmethod
    def from_xgb_booster(booster, max_trees: int) -> "TreeList":
        """
        Load a TreeList from an xgb.Booster object
        Note: We cannot type-hint the xgb.Booster without loading xgb as dependency in pyx code,
              therefore not type hint is added.
        """
        tl = TreeList()
        dump = booster.get_dump(dump_format="json", with_stats=True)
        for tree_id, raw_tree in enumerate(dump):
            if max_trees > 0 and tree_id == max_trees:
                break
            raw_tree_parsed = json.loads(raw_tree)
            root_node = Node.from_xgb_dict(raw_tree_parsed)
            tl.append(TreeView(tree_id=tree_id, root_node=root_node))

        return tl

    @staticmethod
    def from_lightgbm_booster_dump(dump: Dict[str, Any]) -> "TreeList":
        """
        Load a TreeList from a lgbm Booster dump
        Note: We cannot type-hint the the Model without loading lightgbm as dependency in pyx code,
              therefore not type hint is added.
        """
        tl = TreeList()
        for tree_id, tree_dict in enumerate(dump["tree_info"]):
            root_node = Node.from_lightgbm_dict(tree_dict)
            tl.append(TreeView(tree_id=tree_id, root_node=root_node))

        return tl

    def __setitem__(self):
        raise NotImplementedError(
            "Use TreeList.from_*() methods to initialize a TreeList"
        )


def get_lightgbm_params(booster):
    return booster.dump_model()


def get_xgboost_params(booster):
    return json.loads(booster.save_config())


def get_catboost_params(booster):
    with NamedTemporaryFile() as fp:
        booster.save_model(fp.name, "json")
        fp.seek(0)
        model_data = json.load(fp)
    return model_data


def get_gbt_model_from_tree_list(
    tree_list: TreeList,
    n_iterations: int,
    is_regression: bool,
    n_features: int,
    n_classes: int,
    base_score: Optional[float] = None,
):
    """Return a GBT Model from TreeList"""

    if is_regression:
        mb = gbt_reg_model_builder(n_features=n_features, n_iterations=n_iterations)
    else:
        mb = gbt_clf_model_builder(
            n_features=n_features, n_iterations=n_iterations, n_classes=n_classes
        )

    class_label = 0
    for counter, tree in enumerate(tree_list, start=1):
        # find out the number of nodes in the tree
        if is_regression:
            tree_id = mb.create_tree(tree.n_nodes)
        else:
            tree_id = mb.create_tree(n_nodes=tree.n_nodes, class_label=class_label)

        if counter % n_iterations == 0:
            class_label += 1

        if tree.is_leaf:
            mb.add_leaf(tree_id=tree_id, response=tree.value, cover=tree.cover)
            continue

        root_node = tree.root_node
        parent_id = mb.add_split(
            tree_id=tree_id,
            feature_index=root_node.feature,
            feature_value=root_node.get_value_closest_float_downward(),
            cover=root_node.cover,
            default_left=root_node.default_left,
        )

        # create queue
        node_queue: Deque[Node] = deque()
        children = root_node.get_children()
        assert children is not None
        for position, child in enumerate(children):
            child.parent_id = parent_id
            child.position = position
            node_queue.append(child)

        while node_queue:
            node = node_queue.popleft()
            assert node.parent_id != -1, "node.parent_id must not be -1"
            assert node.position != -1, "node.position must not be -1"

            if node.is_leaf:
                mb.add_leaf(
                    tree_id=tree_id,
                    response=node.value,
                    cover=node.cover,
                    parent_id=node.parent_id,
                    position=node.position,
                )
            else:
                parent_id = mb.add_split(
                    tree_id=tree_id,
                    feature_index=node.feature,
                    feature_value=node.get_value_closest_float_downward(),
                    cover=node.cover,
                    default_left=node.default_left,
                    parent_id=node.parent_id,
                    position=node.position,
                )

                children = node.get_children()
                assert children is not None
                for position, child in enumerate(children):
                    child.parent_id = parent_id
                    child.position = position
                    node_queue.append(child)

    return mb.model(base_score=base_score)


def get_gbt_model_from_lightgbm(model: Any, booster=None) -> Any:
    if booster is None:
        booster = model.dump_model()

    n_features = booster["max_feature_idx"] + 1
    n_iterations = len(booster["tree_info"]) / booster["num_tree_per_iteration"]
    n_classes = booster["num_tree_per_iteration"]

    is_regression = False
    objective_fun = booster["objective"]
    if n_classes > 2:
        if "multiclass" not in objective_fun:
            raise TypeError(
                "multiclass (softmax) objective is only supported for multiclass classification"
            )
    elif "binary" in objective_fun:  # nClasses == 1
        n_classes = 2
    else:
        is_regression = True

    tree_list = TreeList.from_lightgbm_booster_dump(booster)

    return get_gbt_model_from_tree_list(
        tree_list,
        n_iterations=n_iterations,
        is_regression=is_regression,
        n_features=n_features,
        n_classes=n_classes,
    )


def get_gbt_model_from_xgboost(booster: Any, xgb_config=None) -> Any:
    # Release Note for XGBoost 1.5.0: Python interface now supports configuring
    # constraints using feature names instead of feature indices. This also
    # helps with pandas input with set feature names.
    booster.feature_names = [str(i) for i in range(booster.num_features())]

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
                "multi:softprob and multi:softmax are only supported for multiclass classification"
            )
    elif objective_fun.startswith("binary:"):
        if objective_fun not in ["binary:logistic", "binary:logitraw"]:
            raise TypeError(
                "only binary:logistic and binary:logitraw are supported for binary classification"
            )
        n_classes = 2
        if objective_fun == "binary:logitraw":
            # daal4py always applies a sigmoid for pred_proba, wheres XGBoost
            # returns raw predictions with logitraw
            warn(
                "objective='binary:logitraw' selected\n"
                "XGBoost returns raw class scores when calling pred_proba()\n"
                "whilst scikit-learn-intelex always uses binary:logistic\n"
            )
            if base_score != 0.5:
                warn("objective='binary:logitraw' ignores base_score, fixing base_score to 0.5")
                base_score = 0.5
    else:
        is_regression = True

    # max_trees=0 if best_iteration does not exist
    max_trees = getattr(booster, "best_iteration", -1) + 1
    if n_classes > 2:
        max_trees *= n_classes
    tree_list = TreeList.from_xgb_booster(booster, max_trees)

    if hasattr(booster, "best_iteration"):
        n_iterations = booster.best_iteration + 1
    else:
        n_iterations = len(tree_list) // (n_classes if n_classes > 2 else 1)

    return get_gbt_model_from_tree_list(
        tree_list,
        n_iterations=n_iterations,
        is_regression=is_regression,
        n_features=n_features,
        n_classes=n_classes,
        base_score=base_score,
    )


def get_gbt_model_from_catboost(model: Any, model_data=None) -> Any:
    if not model.is_fitted():
        raise RuntimeError("Model should be fitted before exporting to daal4py.")

    if model_data is None:
        model_data = get_catboost_params(model)

    if "categorical_features" in model_data["features_info"]:
        raise NotImplementedError(
            "Categorical features are not supported in daal4py Gradient Boosting Trees"
        )

    n_features = len(model_data["features_info"]["float_features"])

    is_symmetric_tree = (
        model_data["model_info"]["params"]["tree_learner_options"]["grow_policy"]
        == "SymmetricTree"
    )

    if is_symmetric_tree:
        n_iterations = len(model_data["oblivious_trees"])
    else:
        n_iterations = len(model_data["trees"])

    n_classes = 0

    if "class_params" in model_data["model_info"]:
        is_classification = True
        n_classes = len(model_data["model_info"]["class_params"]["class_to_label"])
        mb = gbt_clf_model_builder(
            n_features=n_features, n_iterations=n_iterations, n_classes=n_classes
        )
    else:
        is_classification = False
        mb = gbt_reg_model_builder(n_features, n_iterations)

    splits = []

    # Create splits array (all splits are placed sequentially)
    for feature in model_data["features_info"]["float_features"]:
        if feature["borders"]:
            for feature_border in feature["borders"]:
                splits.append(
                    {"feature_index": feature["feature_index"], "value": feature_border}
                )

    if not is_classification:
        bias = model_data["scale_and_bias"][1][0] / n_iterations
        scale = model_data["scale_and_bias"][0]
    else:
        bias = 0
        scale = 1

    trees_explicit = []
    tree_symmetric = []

    if (
        model_data["model_info"]["params"]["data_processing_options"][
            "float_features_binarization"
        ]["nan_mode"]
        == "Min"
    ):
        default_left = 1
    else:
        default_left = 0

    for tree_num in range(n_iterations):
        if is_symmetric_tree:
            if model_data["oblivious_trees"][tree_num]["splits"] is not None:
                # Tree has more than 1 node
                cur_tree_depth = len(model_data["oblivious_trees"][tree_num]["splits"])
            else:
                cur_tree_depth = 0

            tree_symmetric.append(
                (model_data["oblivious_trees"][tree_num], cur_tree_depth)
            )
        else:
            n_nodes = 1
            # Check if node is a leaf (in case of stump)
            if "split" in model_data["trees"][tree_num]:
                # Get number of trees and splits info via BFS
                # Create queue
                nodes_queue = []
                root_node = CatBoostNode(
                    split=splits[model_data["trees"][tree_num]["split"]["split_index"]]
                )
                nodes_queue.append((model_data["trees"][tree_num], root_node))
                while nodes_queue:
                    cur_node_data, cur_node = nodes_queue.pop(0)
                    if "value" in cur_node_data:
                        if isinstance(cur_node_data["value"], list):
                            cur_node.value = [value for value in cur_node_data["value"]]
                        else:
                            cur_node.value = [cur_node_data["value"] * scale + bias]
                    else:
                        cur_node.split = splits[cur_node_data["split"]["split_index"]]
                        left_node = CatBoostNode()
                        right_node = CatBoostNode()
                        cur_node.left = left_node
                        cur_node.right = right_node
                        nodes_queue.append((cur_node_data["left"], left_node))
                        nodes_queue.append((cur_node_data["right"], right_node))
                        n_nodes += 2
            else:
                root_node = CatBoostNode()
                if is_classification and n_classes > 2:
                    root_node.value = [
                        value * scale for value in model_data["trees"][tree_num]["value"]
                    ]
                else:
                    root_node.value = [
                        model_data["trees"][tree_num]["value"] * scale + bias
                    ]
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
                n_nodes = 2 ** (tree_symmetric[i][1] + 1) - 1
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
                cur_tree_leaf_val = cur_tree_info["leaf_values"]
                cur_tree_depth = tree_symmetric[i][1]

                if cur_tree_depth == 0:
                    mb.add_leaf(tree_id=cur_tree_id, response=cur_tree_leaf_val[0])
                else:
                    # One split used for the whole level
                    cur_level_split = splits[
                        cur_tree_info["splits"][cur_tree_depth - 1]["split_index"]
                    ]
                    root_id = mb.add_split(
                        tree_id=cur_tree_id,
                        feature_index=cur_level_split["feature_index"],
                        feature_value=cur_level_split["value"],
                        default_left=default_left,
                        cover=0.0,
                    )
                    prev_level_nodes = [root_id]

                    # Iterate over levels, splits in json are reversed (root split is the last)
                    for cur_level in range(cur_tree_depth - 2, -1, -1):
                        cur_level_nodes = []
                        for cur_parent in prev_level_nodes:
                            cur_level_split = splits[
                                cur_tree_info["splits"][cur_level]["split_index"]
                            ]
                            cur_left_node = mb.add_split(
                                tree_id=cur_tree_id,
                                parent_id=cur_parent,
                                position=0,
                                feature_index=cur_level_split["feature_index"],
                                feature_value=cur_level_split["value"],
                                default_left=default_left,
                                cover=0.0,
                            )
                            cur_right_node = mb.add_split(
                                tree_id=cur_tree_id,
                                parent_id=cur_parent,
                                position=1,
                                feature_index=cur_level_split["feature_index"],
                                feature_value=cur_level_split["value"],
                                default_left=default_left,
                                cover=0.0,
                            )
                            cur_level_nodes.append(cur_left_node)
                            cur_level_nodes.append(cur_right_node)
                        prev_level_nodes = cur_level_nodes

                    # Different storing format for leaves
                    if not is_classification or n_classes == 2:
                        for last_level_node_num in range(len(prev_level_nodes)):
                            mb.add_leaf(
                                tree_id=cur_tree_id,
                                response=cur_tree_leaf_val[2 * last_level_node_num]
                                * scale
                                + bias,
                                parent_id=prev_level_nodes[last_level_node_num],
                                position=0,
                                cover=0.0,
                            )
                            mb.add_leaf(
                                tree_id=cur_tree_id,
                                response=cur_tree_leaf_val[2 * last_level_node_num + 1]
                                * scale
                                + bias,
                                parent_id=prev_level_nodes[last_level_node_num],
                                position=1,
                                cover=0.0,
                            )
                    else:
                        for last_level_node_num in range(len(prev_level_nodes)):
                            left_index = (
                                2 * last_level_node_num * n_tree_each_iter + class_label
                            )
                            right_index = (
                                2 * last_level_node_num + 1
                            ) * n_tree_each_iter + class_label
                            mb.add_leaf(
                                tree_id=cur_tree_id,
                                response=cur_tree_leaf_val[left_index] * scale + bias,
                                parent_id=prev_level_nodes[last_level_node_num],
                                position=0,
                                cover=0.0,
                            )
                            mb.add_leaf(
                                tree_id=cur_tree_id,
                                response=cur_tree_leaf_val[right_index] * scale + bias,
                                parent_id=prev_level_nodes[last_level_node_num],
                                position=1,
                                cover=0.0,
                            )
    else:
        for class_label in range(n_tree_each_iter):
            for i in range(n_iterations):
                root_node = trees_explicit[i][0]

                cur_tree_id = tree_id[i * n_tree_each_iter + class_label]
                # Traverse tree via BFS and build tree with modelbuilder
                if root_node.value is None:
                    root_id = mb.add_split(
                        tree_id=cur_tree_id,
                        feature_index=root_node.split["feature_index"],
                        feature_value=root_node.split["value"],
                        default_left=default_left,
                        cover=0.0,
                    )
                    nodes_queue = [(root_node, root_id)]
                    while nodes_queue:
                        cur_node, cur_node_id = nodes_queue.pop(0)
                        left_node = cur_node.left
                        # Check if node is a leaf
                        if left_node.value is None:
                            left_node_id = mb.add_split(
                                tree_id=cur_tree_id,
                                parent_id=cur_node_id,
                                position=0,
                                feature_index=left_node.split["feature_index"],
                                feature_value=left_node.split["value"],
                                default_left=default_left,
                                cover=0.0,
                            )
                            nodes_queue.append((left_node, left_node_id))
                        else:
                            mb.add_leaf(
                                tree_id=cur_tree_id,
                                response=left_node.value[class_label],
                                parent_id=cur_node_id,
                                position=0,
                                cover=0.0,
                            )
                        right_node = cur_node.right
                        # Check if node is a leaf
                        if right_node.value is None:
                            right_node_id = mb.add_split(
                                tree_id=cur_tree_id,
                                parent_id=cur_node_id,
                                position=1,
                                feature_index=right_node.split["feature_index"],
                                feature_value=right_node.split["value"],
                                default_left=default_left,
                                cover=0.0,
                            )
                            nodes_queue.append((right_node, right_node_id))
                        else:
                            mb.add_leaf(
                                tree_id=cur_tree_id,
                                response=cur_node.right.value[class_label],
                                parent_id=cur_node_id,
                                position=1,
                                cover=0.0,
                            )

                else:
                    # Tree has only one node
                    mb.add_leaf(
                        tree_id=cur_tree_id,
                        response=root_node.value[class_label],
                        cover=0.0,
                    )

    warn("Models converted from CatBoost cannot be used for SHAP value calculation")
    return mb.model(0.0)
