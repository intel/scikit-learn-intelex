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

from pathlib import Path
from typing import Any, Literal, Optional, Protocol, Sequence, TypeVar

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

DType = TypeVar(
    "DType", type(np.float32), type(np.float64), Literal["float32"], Literal["float64"]
)


class ReadCsvFunc(Protocol):
    """Protocol of CSV read functions with the necessary `kwargs`s for testing"""

    def __call__(
        self,
        file_path: Path,
        usecols: Optional[Sequence[int]] = None,
        skip_header: int = 0,
        dtype: DType = np.float32,
        max_rows: Optional[int] = None,
    ) -> Any: ...


class NpReadCsv:
    """Read a CSV using numpy"""

    def __call__(
        self,
        file_path: Path,
        usecols: Optional[Sequence[int]] = None,
        skip_header: int = 0,
        dtype: DType = np.float32,
        max_rows: Optional[int] = None,
    ) -> Any:
        if skip_header is None and max_rows is None:
            if usecols is not None:
                return np.loadtxt(
                    file_path, usecols=usecols, delimiter=",", ndmin=2, dtype=dtype
                )
            else:
                return np.loadtxt(file_path, delimiter=",", ndmin=2, dtype=dtype)

        else:
            a = np.genfromtxt(
                file_path,
                usecols=usecols,
                delimiter=",",
                skip_header=skip_header,
                max_rows=max_rows,
                dtype=dtype,
            )

            if a.shape[0] == 0:
                raise StopIteration("End of file")
            if a.ndim == 1:
                return a[:, np.newaxis]
            return a


class PdReadCsv:
    """Read a CSV using pandas"""

    def __call__(
        self,
        file_path: Path,
        usecols: Optional[Sequence[int]] = None,
        skip_header: int = 0,
        dtype: DType = np.float32,
        max_rows: Optional[int] = None,
    ) -> Any:
        try:
            return pd.read_csv(
                file_path,
                usecols=list(usecols) if usecols is not None else None,
                delimiter=",",
                header=None,
                skiprows=skip_header,
                dtype=dtype,
                nrows=max_rows,
            )
        except pd.errors.EmptyDataError:
            raise StopIteration("End of file")


class CsrReadCsv:
    """Read a CSV and create a scipy CSR matrix"""

    def __call__(
        self,
        file_path: Path,
        usecols: Optional[Sequence[int]] = None,
        skip_header: int = 0,
        dtype: DType = np.float32,
        max_rows: Optional[int] = None,
    ) -> Any:
        data = PdReadCsv()(file_path, usecols, skip_header, dtype, max_rows)
        return csr_matrix(data)


# We have the class definitions to benefit from type safety through protocol classes
# We use the instantiations of these classes to invoke the functionality
np_read_csv = NpReadCsv()
pd_read_csv = PdReadCsv()
csr_read_csv = CsrReadCsv()


def read_next(file_path: Path, chunk_size: int, read_csv: ReadCsvFunc = pd_read_csv):
    end_of_file = False
    skip_header = 0
    while not end_of_file:
        a = read_csv(file_path, skip_header=skip_header, max_rows=chunk_size)
        n_rows = a.shape[0]
        # last chunk is usually smaller, if not,
        # numpy will print warning in next iteration
        if chunk_size > n_rows:
            end_of_file = True
        skip_header += n_rows
        yield a
