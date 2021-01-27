#===============================================================================
# Copyright 2014-2021 Intel Corporation
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

# Example showing reading of file in few chunks, this reader is used in SPMD examples

import numpy as np

# let's try to use pandas' fast csv reader
try:
    import pandas

    def read_csv(f, c=None, sr=0, nr=None, t=np.float64):
        return pandas.read_csv(f, usecols=c, skiprows=sr, nrows=nr,
                               delimiter=',', header=None, dtype=t)
except:
    # fall back to numpy loadtxt
    def read_csv(f, c=None, sr=0, nr=np.iinfo(np.int64).max, t=np.float64):
        res = np.genfromtxt(f, usecols=c, delimiter=',',
                            skip_header=sr, max_rows=nr, dtype=t)
        if res.ndim == 1:
            return res[:, np.newaxis]
        return res


def get_chunk_params(lines_count, chunks_count, chunk_number):
    'returns count of rows to skip from beginning of file and count of rows to read'
    min_nrows = (int)(lines_count / chunks_count)
    rest_rows = lines_count - min_nrows * chunks_count
    is_tail = rest_rows > chunk_number
    skiprows = min_nrows * chunk_number + (chunk_number if is_tail else rest_rows)
    nrows = min_nrows + (1 if is_tail else 0)
    return skiprows, nrows


if __name__ == "__main__":
    infile = "./data/batch/covcormoments_dense.csv"
    chunks_count = 6
    print('Reading file "{}" in {} chunks'.format(infile, chunks_count))

    # Read the whole file to be able to compare
    whole_file = read_csv(infile)

    # Computing chunk-size requires file-size
    lines_in_file = whole_file.shape[0]

    # Read chunks
    chunks_stack = np.empty([0, whole_file.shape[1]])
    for chunk_number in range(chunks_count):
        skiprows, nrows = get_chunk_params(lines_in_file, chunks_count, chunk_number)
        chunk = read_csv(infile, sr=skiprows, nr=nrows)
        print("The shape of chunk number {} is {}".format(chunk_number, chunk.shape))
        chunks_stack = np.vstack((chunks_stack, chunk))

    assert np.array_equal(chunks_stack, whole_file)
    print("Stack of chunks is equal to data in whole file")
