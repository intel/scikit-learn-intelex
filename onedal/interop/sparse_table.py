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

import scipy.sparse as sp

from .csr_table import from_csr_table, is_csr_entity, to_csr_table


def is_sparse_entity(entity) -> bool:
    conditions = [is_csr_entity, sp.isspmatrix]
    return any(map(lambda check: check(entity), conditions))


def to_sparse_table(entity):
    assert is_sparse_entity(entity)
    if sp.isspmatrix(entity):
        entity = entity.tocsr()
    return to_csr_table(entity)


def from_sparse_table(table):
    assert is_sparse_entity(table)
    if is_csr_entity(table):
        result = from_csr_table(table)
    elif sp.isspmatrix(table):
        result = table.to_csr()
    else:
        raise ValueError("Not able to convert from CSR table")
    assert is_csr_entity(table)
    return result
