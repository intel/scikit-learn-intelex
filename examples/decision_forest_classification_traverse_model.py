#*******************************************************************************
# Copyright 2014-2018 Intel Corporation
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
#*******************************************************************************

# daal4py Decision Forest Classification example for shared memory systems

import math    
import daal4py as d4p
from numpy import loadtxt, allclose


def printNodes(node, node_ar, level):
    if not math.isnan(node["threshold"]):
        print(" " * level + "Level " + str(level) + ": Feature = " + str(node["feature"]) + ", threshold = " + str(node["threshold"]))
    if node["left_child"] != -1:
        printNodes(node_ar[node["left_child"]], node_ar, level + 1)
    if node["right_child"] != -1:
        printNodes(node_ar[node["right_child"]], node_ar, level + 1)


if __name__ == "__main__":    
    # input data file
    infile = "./data/batch/df_classification_train.csv"

    # Configure a training object (5 classes)
    train_algo = d4p.decision_forest_classification_training(5, nTrees=10, minObservationsInLeafNode=8, featuresPerNode=3, varImportance="MDI",
                                                             bootstrap=True, resultsToCompute="computeOutOfBagError")

    # Read data. Let"s use 3 features per observation
    data   = loadtxt(infile, delimiter=",", usecols=range(3))
    labels = loadtxt(infile, delimiter=",", usecols=range(3,4))
    labels.shape = (labels.size, 1) # must be a 2d array
    train_result = train_algo.compute(data, labels)

    state = d4p.getTreeState(train_result.model, 0, 5)

    printNodes(state.node_ar[0], state.node_ar, 0)
