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

import os
import subprocess
import unittest
from daal4py.sklearn._utils import get_daal_version
test_path = os.path.abspath(os.path.dirname(__file__))
unittest_data_path = os.path.join(test_path, "unittest_data")
examples_path = os.path.join(
    os.path.dirname(test_path), "examples", "sklearnex")

python_executable = subprocess.run(
    ['/usr/bin/which', 'python'], check=True,
    capture_output=True).stdout.decode().strip()

# First item is major version - 2021,
# second is minor+patch - 0110,
# third item is status - B
sklearnex_version = get_daal_version()
print('oneDAL version:', sklearnex_version)


class TestsklearnexExamples(unittest.TestCase):
    '''Class for testing sklernex examples'''
    # Get a list of all Python files in the examples directory
    pass


def test_generator(file):
    def test(self):
        # Run the script and capture its exit code
        process = subprocess.run(
            [python_executable, os.path.join(examples_path, file)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=True)
        exit_code = process.returncode

        # Assert that the exit code is 0
        self.assertEqual(exit_code, 0)
    return test


if __name__ == '__main__':
    files = [f for f in os.listdir(examples_path) if f.endswith(".py")]
    for file in files:
        test_name = 'test_' + os.path.splitext(file)[0]
        test = test_generator(file)
        setattr(TestsklearnexExamples, test_name, test)
        print("Generating tests for " + test_name)
    unittest.main()
