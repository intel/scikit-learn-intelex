<!--
******************************************************************************
* Copyright 2022 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/-->

# How to Contribute

We welcome community contributions to Intel(R) Extension for Scikit-learn. You can:

- Submit your changes directly with a [pull request](https://github.com/intel/scikit-learn-intelex/pulls).
- Log a bug or make a feature request with an [issue](https://github.com/intel/scikit-learn-intelex/issues).

Refer to our guidelines on [pull requests](#pull-requests) and [issues](#issues) before you proceed.

## Issues

Use [GitHub issues](https://github.com/intel/scikit-learn-intelex/issues) to:
- report an issue
- make a feature request

**Note**: To report a vulnerability, refer to [Intel vulnerability reporting policy](https://www.intel.com/content/www/us/en/security-center/default.html).

## Pull Requests

To contribute your changes directly to the repository, do the following:
- Make sure you can build the product and run all the examples with your patch.
- For a larger feature, provide a relevant example.
- Document your code.
- [Submit](https://github.com/intel/scikit-learn-intelex/pulls) a pull request into the `master` branch. Provide a brief description of the changes you are contributing.

Public CI is enabled for the repository. Your PR should pass all of our checks. We will review your contribution and, if any additional fixes or modifications are necessary, we may give some feedback to guide you. When accepted, your pull request will be merged into our GitHub repository.

## Code Style

We use [black](https://black.readthedocs.io/en/stable/) and [isort](https://pycqa.github.io/isort/) formatters for Python* code. The line length is 90 characters; use default options otherwise. You can find the linter configuration in our [.pyproject.toml](https://github.com/intel/scikit-learn-intelex/blob/master/pyproject.toml).
A GitHub* Action verifies if your changes comply with the output of the auto-formatting tools.

Optionally, you can install pre-commit hooks that do the formatting for you. For this, run from the top level of the repository:

```bash
pip install pre-commit
pre-commit install
```