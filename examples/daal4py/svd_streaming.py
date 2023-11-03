# ==============================================================================
# Copyright 2014 Intel Corporation
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

# daal4py SVD example for streaming on shared memory systems

from pathlib import Path

import daal4py as d4p


def main(*args, **kwargs):
    infiles = [
        Path(__file__).parent / "data" "distributed" / f"svd_{i}.csv" for i in range(1, 5)
    ]

    # configure a SVD object
    algo = d4p.svd(streaming=True)

    # let's provide files directly, not a tables/arrays
    # Feed file by file
    for infile in infiles:
        algo.compute(str(infile))

    # All files are done, now finalize the computation
    result = algo.finalize()

    # SVD result objects provide leftSingularMatrix,
    # rightSingularMatrix and singularValues
    return result


if __name__ == "__main__":
    result = main()
    print("\nSingular values:\n", result.singularValues)
    print("\nRight orthogonal matrix V:\n", result.rightSingularMatrix)
    print(
        "\nLeft orthogonal matrix U (first 10 rows):\n", result.leftSingularMatrix[0:10]
    )
    print("All looks good!")
