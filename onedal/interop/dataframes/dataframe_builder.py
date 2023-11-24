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

import numpy as np

import onedal
from ..array import to_array

from .dtype_conversion import get_data_type
from .column_builder import build_from_column
from .dataframe_protocol import Column, DataFrame

feature_type = onedal._backend.ftype
type_array = onedal._backend.data_management.array_q
metadata = onedal._backend.data_management.table_metadata
heterogen_table = onedal._backend.data_management.heterogen_table

class DataFrameBuilder:
    def __init__(self):
        self.dtypes = []
        self.columns = []

    @property
    def column_count(self) -> int:
        result = len(self.columns)
        control = len(self.dtypes)
        assert result == control
        return result

    def append(self, column: Column):
        self.dtypes.append(column.dtype)
        self.columns.append(column)
        return self

    def build_dtype_array(self) -> type_array:
        result = list()
        for index in range(self.column_count):
            dtype = self.dtypes[index]
            dal_dtype = get_data_type(dtype)
            result.append(dal_dtype)
        result = np.asarray(result)
        result = result.astype(np.int32)
        return to_array(result)
    
    # TODO: implement logic supporting 
    def build_ftype_array(self) -> type_array:
        ratio = feature_type.ratio
        col_count = self.column_count
        result = np.full(col_count, ratio)
        result = result.astype(np.int32)
        return to_array(result)
    
    def build_metadata(self) -> metadata:
        dtypes = self.build_dtype_array()
        ftypes = self.build_ftype_array()
        return metadata(dtypes, ftypes)
    
    def __validate(self, result: heterogen_table):
        column_count = result.get_column_count()
        assert column_count == self.column_count
    
    def build(self) -> heterogen_table:
        meta = self.build_metadata()
        result = heterogen_table(meta)


        for index in range(self.column_count):
            column = self.columns[index]
            array = build_from_column(column)
            result.set_column(index, array)

        self.__validate(result)
        return result
    
def build_from_dataframe(df: DataFrame) -> heterogen_table:
    column_count = df.num_columns()
    builder = DataFrameBuilder()

    for index in range(column_count):
        column = df.get_column(index)
        builder.append(column)

    result = builder.build()
    row_count = result.get_row_count()
    assert row_count == df.num_rows()
    return result
