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

import os
import subprocess
import sys
import unittest

from daal4py.sklearn._utils import get_daal_version

test_path = os.path.abspath(os.path.dirname(__file__))
unittest_data_path = os.path.join(test_path, "unittest_data")
examples_path = os.path.join(os.path.dirname(test_path), "examples", "sklearnex")

print("Testing examples_sklearnex")
# First item is major version - 2021,
# second is minor+patch - 0110,
# third item is status - B
sklearnex_version = get_daal_version()
print("oneDAL version:", sklearnex_version)


class TestsklearnexExamples(unittest.TestCase):
    """Class for testing sklernex examples"""

    # Get a list of all Python files in the examples directory
    pass


def test_generator(file):
    def testit(self):
        # Run the script and capture its exit code
        process = subprocess.run(
            [sys.executable, os.path.join(examples_path, file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )  # nosec
        exit_code = process.returncode

        # Assert that the exit code is 0
        self.assertEqual(
            exit_code,
            0,
            msg=f"Example has failed, the example's output:\n{process.stdout.decode()}\n{process.stderr.decode()}",
        )

    setattr(TestsklearnexExamples, "test_" + os.path.splitext(file)[0], testit)
    print("Generating tests for " + os.path.splitext(file)[0])


files = [
    f
    for f in os.listdir(examples_path)
    if f.endswith(".py") and "spmd" not in f and "dpnp" not in f and "dpctl" not in f
]

for file in files:
    test_generator(file)

if __name__ == "__main__":
    unittest.main()
