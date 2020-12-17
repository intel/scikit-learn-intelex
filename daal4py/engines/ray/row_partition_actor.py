#
#*******************************************************************************
# Copyright 2014-2020 Intel Corporation
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
#******************************************************************************/

import pandas

class RowPartitionsActor:
    def __init__(self, node):
        self.row_parts_ = []
        self.node_ = node

    def set_row_parts(self, *row_parts):
        self.row_parts_ = pandas.concat(list(row_parts), axis=0)

    def set_row_parts_list(self, *row_parts_list):
        self.row_parts_ = list(row_parts_list)

    def append_row_part(self, row_part):
        self.row_parts_.append(row_part)

    def concat_row_parts(self):
        self.row_parts_ = pandas.concat(self.row_parts_, axis=0)

    def get_row_parts(self):
        return self.row_parts_

    def get_actor_ip(self):
        return self.node_
