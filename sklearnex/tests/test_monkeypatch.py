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

import sklearnex


def test_monkey_patching():
    _tokens = sklearnex.get_patch_names()
    _values = sklearnex.get_patch_map().values()
    _classes = list()

    for v in _values:
        for c in v:
            _classes.append(c[0])

    sklearnex.patch_sklearn()

    for i, _ in enumerate(_tokens):
        t = _tokens[i]
        p = _classes[i][0]
        n = _classes[i][1]

        class_module = getattr(p, n).__module__
        assert \
            class_module.startswith('daal4py') or class_module.startswith('sklearnex'), \
            "Patching has completed with error."

    for i, _ in enumerate(_tokens):
        t = _tokens[i]
        p = _classes[i][0]
        n = _classes[i][1]

        sklearnex.unpatch_sklearn(t)
        class_module = getattr(p, n).__module__
        assert class_module.startswith('sklearn'), \
            "Unpatching has completed with error."

    sklearnex.unpatch_sklearn()

    for i, _ in enumerate(_tokens):
        t = _tokens[i]
        p = _classes[i][0]
        n = _classes[i][1]

        class_module = getattr(p, n).__module__
        assert class_module.startswith('sklearn'), \
            "Unpatching has completed with error."

    sklearnex.unpatch_sklearn()

    for i, _ in enumerate(_tokens):
        t = _tokens[i]
        p = _classes[i][0]
        n = _classes[i][1]

        sklearnex.patch_sklearn(t)

        class_module = getattr(p, n).__module__
        assert \
            class_module.startswith('daal4py') or class_module.startswith('sklearnex'), \
            "Patching has completed with error."

    sklearnex.unpatch_sklearn()


def test_patch_by_list_simple():
    sklearnex.patch_sklearn(["LogisticRegression"])

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    assert RandomForestRegressor.__module__.startswith('sklearn')
    assert KNeighborsRegressor.__module__.startswith('sklearn')
    assert LogisticRegression.__module__.startswith('daal4py')
    assert SVC.__module__.startswith('sklearn')

    sklearnex.unpatch_sklearn()


def test_patch_by_list_many_estimators():
    sklearnex.patch_sklearn(["LogisticRegression", "SVC"])

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    assert RandomForestRegressor.__module__.startswith('sklearn')
    assert KNeighborsRegressor.__module__.startswith('sklearn')
    assert LogisticRegression.__module__.startswith('daal4py')
    assert SVC.__module__.startswith('daal4py') or SVC.__module__.startswith('sklearnex')

    sklearnex.unpatch_sklearn()


def test_unpatch_by_list_many_estimators():
    sklearnex.patch_sklearn()

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    assert RandomForestRegressor.__module__.startswith('sklearnex')
    assert KNeighborsRegressor.__module__.startswith('daal4py') or \
        KNeighborsRegressor.__module__.startswith('sklearnex')
    assert LogisticRegression.__module__.startswith('daal4py')
    assert SVC.__module__.startswith('daal4py') or SVC.__module__.startswith('sklearnex')

    sklearnex.unpatch_sklearn(["KNeighborsRegressor", "RandomForestRegressor"])

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    assert RandomForestRegressor.__module__.startswith('sklearn')
    assert KNeighborsRegressor.__module__.startswith('sklearn')
    assert LogisticRegression.__module__.startswith('daal4py')
    assert SVC.__module__.startswith('daal4py') or SVC.__module__.startswith('sklearnex')


def test_patching_checker():
    for name in [None, 'SVC', 'PCA']:
        sklearnex.patch_sklearn(name=name)
        assert sklearnex.sklearn_is_patched(name=name)

        sklearnex.unpatch_sklearn(name=name)
        assert not sklearnex.sklearn_is_patched(name=name)

    sklearnex.patch_sklearn()
    patching_status_map = sklearnex.sklearn_is_patched(return_map=True)
    assert len(patching_status_map) == len(sklearnex.get_patch_names())
    for status in patching_status_map.values():
        assert status

    sklearnex.unpatch_sklearn()
    patching_status_map = sklearnex.sklearn_is_patched(return_map=True)
    assert len(patching_status_map) == len(sklearnex.get_patch_names())
    for status in patching_status_map.values():
        assert not status
