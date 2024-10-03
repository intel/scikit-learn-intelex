#!/bin/bash
#===============================================================================
# Copyright 2021 Intel Corporation
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

SAMPLES_DIR=sources/samples

# remove the samples folder if it exists
if [ -d "$SAMPLES_DIR" ]; then rm -Rf $SAMPLES_DIR; fi

# create a samples folder
mkdir $SAMPLES_DIR

# copy jupyter notebooks
cd ..
cp examples/notebooks/*.ipynb doc/$SAMPLES_DIR

# to build the docs locally, uncomment the following lines
# cd doc
# make html
