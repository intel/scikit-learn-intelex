#===============================================================================
# Copyright 2021 Intel Corporation
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

import re
import subprocess
import sys
import pytest


def get_method(s):
    return s.split('.')[3].split(':')[0]


def get_branch(s):
    if len(s) == 0:
        return 'NO INFO'
    for i in s:
        if 'uses original Scikit-learn solver,' in i:
            return 'was in OPT, but go in Scikit'
    for i in s:
        if 'uses Intel(R) oneAPI Data Analytics Library solver' in i:
            return 'OPT'
    return 'Scikit'


def run_parse(mas, result):
    name, dtype = mas[0].split()
    temp = []
    for i in range(1, len(mas)):
        mas[i] = mas[i][6:]
        if not mas[i].startswith('sklearn'):
            ind = name + ' ' + dtype + ' ' + mas[i]
            result[ind] = get_branch(temp)
            temp.clear()
        else:
            temp.append(mas[i])


TO_SKIP = [
    # --------------- NO INFO ---------------
    r'KMeans .*transform',
    r'KMeans .*score',
    r'PCA .*score',
    r'LogisticRegression .*decision_function',
    r'LogisticRegressionCV .*decision_function',
    r'LogisticRegressionCV .*predict',
    r'LogisticRegressionCV .*predict_proba',
    r'LogisticRegressionCV .*predict_log_proba',
    r'LogisticRegressionCV .*score',
    # --------------- Scikit ---------------
    r'Ridge float16 predict',
    r'Ridge float16 score',
    r'RandomForestClassifier .*predict_proba',
    r'RandomForestClassifier .*predict_log_proba',
    r'pairwise_distances .*pairwise_distances',  # except float64
    r'roc_auc_score .*roc_auc_score',  # except float32 and float64
]


def get_result_log():
    process = subprocess.run(
        [
            sys.executable,
            'daal4py/sklearn/test/test_patching/launch_algorithms.py'
        ],
        capture_output=True, text=True
    )
    mas = []
    result = {}
    for i in process.stdout.split('\n'):
        if not i.startswith('INFO') and len(mas) != 0:
            run_parse(mas, result)
            mas.clear()
            mas.append(i.strip())
        else:
            mas.append(i.strip())
    return result


result_log = get_result_log()


@pytest.mark.parametrize('configuration', result_log)
def test_patching(configuration):
    if 'OPT' in result_log[configuration]:
        return
    for skip in TO_SKIP:
        if re.search(skip, configuration) is not None:
            pytest.skip("SKIPPED", allow_module_level=False)
    raise ValueError('Test patching failed: ' + configuration)
