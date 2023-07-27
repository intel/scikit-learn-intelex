# ===============================================================================
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
# ===============================================================================

import json
from collections import deque
from dataclasses import dataclass
from os import getpid, remove
from time import time
from typing import TYPE_CHECKING, Any, Deque, Dict, Generator, List, Optional

import numpy as np

if TYPE_CHECKING:
    import xgboost as xgb


@dataclass
class Node:
    """Helper class holding Tree Node information"""

    tree_id: int
    node_id: int
    left_child_id: Optional[int]
    right_child_id: Optional[int]
    cover: float
    is_leaf: bool
    default_left: bool
    feature: Optional[int]
    value: Optional[float]
    parent_id: Optional[int] = -1
    position: Optional[int] = -1

    def get_value_closest_float_downward(self) -> np.float64:
        """Get the closest exact fp value smaller than self.value"""
        return np.nextafter(np.single(self.value), np.single(-np.inf))


class TreeView:
    """Helper class, treating a list of nodes as one tree"""

    def __init__(self, tree_id: int, nodes: list[Node]) -> None:
        self.tree_id = tree_id
        self.nodes = nodes
        self.n_nodes = len(nodes)

    @property
    def is_leaf(self) -> bool:
        return len(self.nodes) == 1 and self.nodes[0].is_leaf

    @property
    def value(self) -> float:
        if not self.is_leaf:
            raise ValueError("Tree is not a leaf-only tree")
        if not self.nodes[0].value:
            raise ValueError("Tree is leaf-only but leaf node has no value")
        return self.nodes[0].value

    def get_children(self, node: Node) -> tuple[Node, Node]:
        """Find children of the provided node"""
        children_ids = (node.left_child_id, node.right_child_id)
        selection = [n for n in self.nodes if n.node_id in children_ids]
        assert (
            len(selection) == 2
        ), f"Found {len(selection)} (!= 2) child nodes for node {node}"
        return tuple(selection)


class NodeList(list):
    """Helper class that is able to extract all information required by the
    model builders from an XGBoost.Booster object"""

    @staticmethod
    def from_booster(booster: xgb.Booster) -> "NodeList":
        """Create a TreeList object from a xgb.Booster object"""
        tl = NodeList()
        df = booster.trees_to_dataframe()
        for _, node in df.iterrows():
            tree_id, node_id = map(int, node["ID"].split("-"))  # e.g. 0-1
            is_leaf = node["Feature"] == "Leaf"
            left_child_id = (
                int(node["Yes"].split("-")[1]) if isinstance(node["Yes"], str) else None
            )
            right_child_id = (
                int(node["No"].split("-")[1]) if isinstance(node["No"], str) else None
            )
            tl.append(
                Node(
                    tree_id=tree_id,
                    node_id=node_id,
                    left_child_id=left_child_id,
                    right_child_id=right_child_id,
                    cover=node["Cover"],
                    feature=int(node["Feature"]) if node["Feature"].isnumeric() else None,
                    is_leaf=is_leaf,
                    default_left=node["Yes"] == node["Missing"],
                    value=None if is_leaf else node["Split"],
                )
            )

        # fill the missing leaf values which are not part of the dataframe
        tl._fill_leaf_values(booster.get_dump(dump_format="json"))

        return tl

    def iter_trees(self) -> Generator[TreeView, None, None]:
        """Iterate over TreeViews"""
        tree_ids = set((node.tree_id for node in self))
        for tid in tree_ids:
            yield TreeView(tree_id=tid, nodes=[n for n in self if n.tree_id == tid])

    def _fill_leaf_values(self, booster_dump: list[str]) -> None:
        """Fill the leaf values (i.e. the predictions) from `booster_dump`
        Note: These values are not contained in the pd.DataFrame format"""

        def get_leaf_nodes(
            node: Dict[str, Any], leaf_nodes: list[Dict[str, Any]] = []
        ) -> None:
            """Helper to get all leaf nodes from the json.loads() of the booster_dump"""
            if "children" in node:
                get_leaf_nodes(node["children"][0], leaf_nodes)
                get_leaf_nodes(node["children"][1], leaf_nodes)
                return

            if "leaf" not in node:
                raise KeyError(f"Node does not have a 'leaf' value: {node}")

            leaf_nodes.append(node)

        root_nodes = [json.loads(s) for s in booster_dump]

        for tree_id, root_node in enumerate(root_nodes):
            leaf_nodes = []
            get_leaf_nodes(root_node, leaf_nodes)

            for node in self:
                if not node.is_leaf:
                    continue

                if node.tree_id != tree_id:
                    continue

                try:
                    node.value = float(
                        [
                            l["leaf"] for l in leaf_nodes if l["nodeid"] == node.node_id
                        ].pop()
                    )
                except IndexError as e:
                    raise ValueError(
                        f"No leaf information for node {node.node_id} in tree {node.tree_id}"
                    ) from e

        # assert all tree leafs have a value
        for node in self:
            if node.is_leaf:
                assert (
                    node.value is not None
                ), f"Failed to find leaf value for node {node}"

    def __setitem__(self):
        raise NotImplementedError("Use TreeList.from_booster() to initialize a TreeList")


def get_lightgbm_params(booster):
    return booster.dump_model()


def get_xgboost_params(booster):
    return json.loads(booster.save_config())


def get_catboost_params(booster):
    dump_filename = f"catboost_model_{getpid()}_{time()}"

    # Dump model in file
    booster.save_model(dump_filename, "json")

    # Read json with model
    with open(dump_filename) as file:
        model_data = json.load(file)

    # Delete dump file
    remove(dump_filename)
    return model_data


def get_gbt_model_from_lightgbm(model: Any, lgb_model=None) -> Any:
    @dataclass
    class LightGbmNode:
        tree: Dict[str, Any]
        parent_id: int
        position: int

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
                "multiclass (softmax) objective is only supported for multiclass classification"
            )
    elif "binary" in objective_fun:  # nClasses == 1
        n_classes = 2
    else:
        is_regression = True

    if is_regression:
        mb = gbt_reg_model_builder(n_features=n_features, n_iterations=n_iterations)
    else:
        mb = gbt_clf_model_builder(
            n_features=n_features, n_iterations=n_iterations, n_classes=n_classes
        )

    class_label = 0
    iterations_counter = 0
    for tree in lgb_model["tree_info"]:
        if is_regression:
            tree_id = mb.create_tree(tree["num_leaves"] * 2 - 1)
        else:
            tree_id = mb.create_tree(
                n_nodes=tree["num_leaves"] * 2 - 1, class_label=class_label
            )

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
                "Categorical features are not supported in daal4py Gradient Boosting Trees"
            )
        default_left = int(sub_tree["default_left"])
        parent_id = mb.add_split(
            tree_id=tree_id,
            feature_index=sub_tree["split_feature"],
            feature_value=feat_val,
            default_left=default_left,
        )

        # create stack
        node_stack: List[LightGbmNode] = [
            LightGbmNode(sub_tree["left_child"], parent_id, 0),
            LightGbmNode(sub_tree["right_child"], parent_id, 1),
        ]

        # dfs through it
        while node_stack:
            sub_tree = node_stack[-1].tree
            parent_id = node_stack[-1].parent_id
            position = node_stack[-1].position
            node_stack.pop()

            # current node is leaf
            if "leaf_index" in sub_tree:
                mb.add_leaf(
                    tree_id=tree_id,
                    response=sub_tree["leaf_value"],
                    parent_id=parent_id,
                    position=position,
                )
                continue

            # current node is split
            feat_val = sub_tree["threshold"]
            if isinstance(feat_val, str):
                raise NotImplementedError(
                    "Categorical features are not supported in daal4py Gradient Boosting Trees"
                )
            default_left = int(sub_tree["default_left"])
            parent_id = mb.add_split(
                tree_id=tree_id,
                feature_index=sub_tree["split_feature"],
                feature_value=feat_val,
                default_left=default_left,
                parent_id=parent_id,
                position=position,
            )

            # append children
            node_stack.append(LightGbmNode(sub_tree["left_child"], parent_id, 0))
            node_stack.append(LightGbmNode(sub_tree["right_child"], parent_id, 1))

    return mb.model()


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
    elif objective_fun.find("binary:") == 0:
        if objective_fun in ["binary:logistic", "binary:logitraw"]:
            n_classes = 2
        else:
            raise TypeError(
                "binary:logistic and binary:logitraw are only supported for binary classification"
            )
    else:
        is_regression = True

    n_iterations = booster.best_iteration + 1

    # Create + base iteration
    if is_regression:
        mb = gbt_reg_model_builder(n_features=n_features, n_iterations=n_iterations + 1)

        tree_id = mb.create_tree(1)
        mb.add_leaf(tree_id=tree_id, response=base_score)
    else:
        mb = gbt_clf_model_builder(
            n_features=n_features, n_iterations=n_iterations, n_classes=n_classes
        )

    class_label = 0
    node_list = NodeList.from_booster(booster)
    for counter, tree in enumerate(node_list.iter_trees(), start=1):
        # find out the number of nodes in the tree
        if is_regression:
            tree_id = mb.create_tree(tree.n_nodes)
        else:
            tree_id = mb.create_tree(n_nodes=tree.n_nodes, class_label=class_label)

        if counter % n_iterations == 0:
            class_label += 1

        if tree.is_leaf:
            mb.add_leaf(tree_id=tree_id, response=tree.value)
            continue

        root_node = tree.nodes[0]
        assert isinstance(
            root_node.feature, int
        ), f"Feature names must be integers (got ({type(root_node.feature)}){root_node.feature})"
        parent_id = mb.add_split(
            tree_id=tree_id,
            feature_index=root_node.feature,
            feature_value=root_node.get_value_closest_float_downward(),
            default_left=root_node.default_left,
        )

        # create queue
        node_queue: Deque[Node] = deque()
        children = tree.get_children(root_node)
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
                    parent_id=node.parent_id,
                    position=node.position,
                )
            else:
                assert isinstance(
                    node.feature, int
                ), f"Feature names must be integers (got ({type(node.feature)}){node.feature})"
                parent_id = mb.add_split(
                    tree_id=tree_id,
                    feature_index=node.feature,
                    feature_value=node.get_value_closest_float_downward(),
                    default_left=node.default_left,
                    parent_id=node.parent_id,
                    position=node.position,
                )

                children = tree.get_children(node)
                for position, child in enumerate(children):
                    child.parent_id = parent_id
                    child.position = position
                    node_queue.append(child)

    return mb.model()


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

            @dataclass
            class CatBoostNode:
                split: Optional[float] = None
                value: Optional[list[float]] = None
                right: Optional[int] = None
                left: Optional[int] = None

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
                            )
                            cur_right_node = mb.add_split(
                                tree_id=cur_tree_id,
                                parent_id=cur_parent,
                                position=1,
                                feature_index=cur_level_split["feature_index"],
                                feature_value=cur_level_split["value"],
                                default_left=default_left,
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
                            )
                            mb.add_leaf(
                                tree_id=cur_tree_id,
                                response=cur_tree_leaf_val[2 * last_level_node_num + 1]
                                * scale
                                + bias,
                                parent_id=prev_level_nodes[last_level_node_num],
                                position=1,
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
                            )
                            mb.add_leaf(
                                tree_id=cur_tree_id,
                                response=cur_tree_leaf_val[right_index] * scale + bias,
                                parent_id=prev_level_nodes[last_level_node_num],
                                position=1,
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
                            )
                            nodes_queue.append((left_node, left_node_id))
                        else:
                            mb.add_leaf(
                                tree_id=cur_tree_id,
                                response=left_node.value[class_label],
                                parent_id=cur_node_id,
                                position=0,
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
                            )
                            nodes_queue.append((right_node, right_node_id))
                        else:
                            mb.add_leaf(
                                tree_id=cur_tree_id,
                                response=cur_node.right.value[class_label],
                                parent_id=cur_node_id,
                                position=1,
                            )

                else:
                    # Tree has only one node
                    mb.add_leaf(
                        tree_id=cur_tree_id, response=root_node.value[class_label]
                    )

    return mb.model()
