# ==============================================================================
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
# ==============================================================================

import importlib.util
import unittest

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import daal4py as d4p
from daal4py import _get__daal_link_version__ as dv
from daal4py.sklearn._utils import daal_check_version

# First item is major version - 2021,
# second is minor+patch - 0110,
# third item is status - B
daal_version = (int(dv()[0:4]), dv()[10:11], int(dv()[4:8]))
reason = str(((2021, "P", 1))) + " not supported in this library version "
reason += str(daal_version)


class XgboostModelBuilder(unittest.TestCase):
    # @unittest.skipUnless(
    #     all(
    #         [
    #             hasattr(d4p, "get_gbt_model_from_xgboost"),
    #             hasattr(d4p, "gbt_classification_prediction"),
    #             daal_check_version(((2021, "P", 1))),
    #         ]
    #     ),
    #     reason,
    # )
    # @unittest.skipUnless(
    #     importlib.util.find_spec("xgboost") is not None,
    #     "xgboost library is not installed",
    # )
    # def test_earlystop(self):
    #     import xgboost as xgb

    #     num_classes = 3
    #     X, y = make_classification(
    #         n_samples=1000,
    #         n_features=10,
    #         n_informative=3,
    #         n_classes=num_classes,
    #         random_state=42,
    #     )
    #     X_train, X_test, y_train, y_test = train_test_split(
    #         X, y, test_size=0.3, random_state=42
    #     )

    #     # training parameters setting
    #     params = {
    #         "n_estimators": 100,
    #         "max_bin": 256,
    #         "scale_pos_weight": 2,
    #         "lambda_l2": 1,
    #         "alpha": 0.9,
    #         "max_depth": 8,
    #         "num_leaves": 2**8,
    #         "verbosity": 0,
    #         "objective": "multi:softproba",
    #         "learning_rate": 0.3,
    #         "num_class": num_classes,
    #         "early_stopping_rounds": 5,
    #     }

    #     xgb_clf = xgb.XGBClassifier(**params)
    #     xgb_clf.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    #     booster = xgb_clf.get_booster()

    #     xgb_prediction = xgb_clf.predict(X_test)
    #     xgb_proba = xgb_clf.predict_proba(X_test)
    #     xgb_errors_count = np.count_nonzero(xgb_prediction - np.ravel(y_test))

    #     daal_model = d4p.mb.convert_model(booster)

    #     daal_prediction = daal_model.predict(X_test)
    #     daal_proba = daal_model.predict_proba(X_test)
    #     daal_errors_count = np.count_nonzero(daal_prediction - np.ravel(y_test))

    #     self.assertTrue(np.absolute(xgb_errors_count - daal_errors_count) == 0)
    #     self.assertTrue(np.allclose(xgb_proba, daal_proba))

    @unittest.skipUnless(
        all(
            [
                hasattr(d4p, "get_gbt_model_from_xgboost"),
                hasattr(d4p, "gbt_classification_prediction"),
                daal_check_version(((2021, "P", 1))),
            ]
        ),
        reason,
    )
    @unittest.skipUnless(
        importlib.util.find_spec("xgboost") is not None,
        "xgboost library is not installed",
    )
    def test_model_from_booster(self):
        class MockBooster:
            def get_dump(self, *_, **kwargs):
                # raw dump of 2 trees with a max depth of 1
                return [
                    '  { "nodeid": 0, "depth": 0, "split": "1", "split_condition": 2, "yes": 1, "no": 2, "missing": 1 , "gain": 3, "cover": 4, "children": [\n    { "nodeid": 1, "leaf": 5 , "cover": 6 }, \n    { "nodeid": 2, "leaf": 7 , "cover":8 }\n  ]}',
                    '  { "nodeid": 0, "leaf": 0.2 , "cover": 42 }',
                ]

        mock = MockBooster()
        result = d4p.TreeList.from_booster(mock)
        self.assertEqual(len(result), 2)

        tree0 = result[0]
        self.assertIsInstance(tree0, d4p.TreeView)
        self.assertFalse(tree0.is_leaf)
        with self.assertRaises(ValueError):
            tree0.cover
        with self.assertRaises(ValueError):
            tree0.value

        self.assertIsInstance(tree0.root_node, d4p.Node)

        self.assertEqual(tree0.root_node.node_id, 0)
        self.assertEqual(tree0.root_node.left_child.node_id, 1)
        self.assertEqual(tree0.root_node.right_child.node_id, 2)

        self.assertEqual(tree0.root_node.cover, 4)
        self.assertEqual(tree0.root_node.left_child.cover, 6)
        self.assertEqual(tree0.root_node.right_child.cover, 8)

        self.assertFalse(tree0.root_node.is_leaf)
        self.assertTrue(tree0.root_node.left_child.is_leaf)
        self.assertTrue(tree0.root_node.right_child.is_leaf)

        self.assertTrue(tree0.root_node.default_left)
        self.assertFalse(tree0.root_node.left_child.default_left)
        self.assertFalse(tree0.root_node.right_child.default_left)

        self.assertEqual(tree0.root_node.feature, 1)
        with self.assertRaises(ValueError):
            tree0.root_node.left_child.feature
        with self.assertRaises(ValueError):
            tree0.root_node.right_child.feature

        self.assertEqual(tree0.root_node.value, 2)
        self.assertEqual(tree0.root_node.left_child.value, 5)
        self.assertEqual(tree0.root_node.right_child.value, 7)

        self.assertEqual(tree0.root_node.n_children, 2)
        self.assertEqual(tree0.root_node.left_child.n_children, 0)
        self.assertEqual(tree0.root_node.right_child.n_children, 0)

        self.assertIsNone(tree0.root_node.left_child.left_child)
        self.assertIsNone(tree0.root_node.left_child.right_child)
        self.assertIsNone(tree0.root_node.right_child.left_child)
        self.assertIsNone(tree0.root_node.right_child.right_child)

        tree1 = result[1]
        self.assertIsInstance(tree1, d4p.TreeView)
        self.assertTrue(tree1.is_leaf)
        self.assertEqual(tree1.n_nodes, 1)
        self.assertEqual(tree1.cover, 42)
        self.assertEqual(tree1.value, 0.2)


if __name__ == "__main__":
    unittest.main()
