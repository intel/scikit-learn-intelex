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

import importlib
import inspect
import os
import sys
import time
import unittest
from pathlib import Path
from types import ModuleType
from typing import Callable, Iterator, Optional

import numpy as np

from daal4py.sklearn._utils import get_daal_version
from daal4py.sklearn.utils import ReadCsvFunc, csr_read_csv, np_read_csv, pd_read_csv
from daal4py.tests.config import Config, Daal4pyExampleConfig

project_dir = Path(__file__).parent.parent.parent
unittest_data_dir = project_dir / "tests" / "unittest_data"


class TestCaseFactory:
    class TestCaseForDaal4pyExample(unittest.TestCase):
        """Template for the test case - to be populated by the factory"""

        _module: Optional[ModuleType] = None

    _example_path = project_dir / "examples" / "daal4py"
    _example_config = Config(project_dir / "examples" / "tests" / "test_daal4py.yml")

    @staticmethod
    def _load_module(path: Path) -> ModuleType:
        import_path = str(path.parent)
        if import_path not in sys.path:
            sys.path.insert(0, import_path)
        return importlib.import_module(path.stem)

    @staticmethod
    def _gen_function(
        has_main_function: bool,
        example_config: Daal4pyExampleConfig,
        accepts_read_csv_argument: bool,
        read_csv: ReadCsvFunc = np_read_csv,
        timeout_cpu_seconds: int = 90,
    ) -> Callable[["TestCaseForDaal4pyExample"], None]:
        version_check, version_reason = example_config.check_version(get_daal_version())
        dependency_check, dependency_reason = example_config.check_dependencies()

        @unittest.skipUnless(has_main_function, f"Module has no main function")
        @unittest.skipUnless(version_check, version_reason)
        @unittest.skipUnless(dependency_check, dependency_reason)
        def test_case(self: TestCaseFactory.TestCaseForDaal4pyExample):
            assert self._module is not None

            kwargs = dict()
            if accepts_read_csv_argument:
                kwargs["readcsv"] = read_csv

            start = time.process_time()
            results = self._module.main(**kwargs)
            if results and example_config.results_csv:
                reference = np_read_csv(unittest_data_dir / example_config.results_csv)
                np.testing.assert_allclose(results, reference)
            duration_seconds = time.process_time() - start

            self.assertLessEqual(duration_seconds, timeout_cpu_seconds)

        return test_case

    @staticmethod
    def load_to_namespace(path: Path) -> None:
        module = TestCaseFactory._load_module(path)

        # Create a new class for the requested module
        class TestClass(TestCaseFactory.TestCaseForDaal4pyExample):
            _module = module

        has_main_function = hasattr(module, "main")
        if has_main_function:
            signature = inspect.signature(module.main)
            args = list(signature.parameters)
            accepts_read_csv_argument = "readcsv" in args
        else:
            accepts_read_csv_argument = False

        config = TestCaseFactory._example_config[path.stem]
        assert isinstance(config, Daal4pyExampleConfig)

        gen_kwargs = {
            "has_main_function": has_main_function,
            "example_config": config,
            "accepts_read_csv_argument": accepts_read_csv_argument,
            "read_csv": np_read_csv,
        }

        # Populate the default test case
        setattr(
            TestClass,
            "test_np_read_csv",
            TestCaseFactory._gen_function(**gen_kwargs),
        )
        if accepts_read_csv_argument:
            # run with remaining `ReadCsvFunc`s
            gen_kwargs["read_csv"] = pd_read_csv
            setattr(
                TestClass,
                "test_np_read_csv",
                TestCaseFactory._gen_function(**gen_kwargs),
            )

            gen_kwargs["read_csv"] = csr_read_csv
            setattr(
                TestClass,
                "test_csr_read_csv",
                TestCaseFactory._gen_function(**gen_kwargs),
            )

        # Register the class in the global name space for unittest discovery
        test_name = f"test_{path.stem}"
        globals()[test_name] = type(test_name, (TestClass,), {})

    @staticmethod
    def get_all_source_files() -> Iterator[Path]:
        for filename in os.listdir(str(TestCaseFactory._example_path.resolve())):
            if filename.endswith(".py"):
                yield TestCaseFactory._example_path / filename


for path in TestCaseFactory.get_all_source_files():
    TestCaseFactory.load_to_namespace(path)

if __name__ == "__main__":
    unittest.main()
