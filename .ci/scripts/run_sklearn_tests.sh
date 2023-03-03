#!/bin/bash
#===============================================================================
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
#===============================================================================

ci_dir=$( dirname $( dirname "${BASH_SOURCE[0]}" ) )
cd $ci_dir

export SELECTED_TESTS=$(python scripts/select_sklearn_tests.py)
export DESELECTED_TESTS=$(python ../.circleci/deselect_tests.py deselected_tests.yaml)
cd $(python -c "import sklearn, os; print(os.path.dirname(sklearn.__file__))")
export SKLEARNEX_VERBOSE=DEBUG
python -m sklearnex -m pytest --verbose --pyargs --durations=100 --durations-min=0.01 $DESELECTED_TESTS $SELECTED_TESTS
