# ===============================================================================
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
# ===============================================================================

import sklearnex
from daal4py.sklearn._utils import daal_check_version


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
        assert class_module.startswith("daal4py") or class_module.startswith(
            "sklearnex"
        ), "Patching has completed with error."

    for i, _ in enumerate(_tokens):
        t = _tokens[i]
        p = _classes[i][0]
        n = _classes[i][1]

        sklearnex.unpatch_sklearn(t)
        class_module = getattr(p, n).__module__
        assert class_module.startswith("sklearn"), "Unpatching has completed with error."

    sklearnex.unpatch_sklearn()

    for i, _ in enumerate(_tokens):
        t = _tokens[i]
        p = _classes[i][0]
        n = _classes[i][1]

        class_module = getattr(p, n).__module__
        assert class_module.startswith("sklearn"), "Unpatching has completed with error."

    sklearnex.unpatch_sklearn()

    for i, _ in enumerate(_tokens):
        t = _tokens[i]
        p = _classes[i][0]
        n = _classes[i][1]

        sklearnex.patch_sklearn(t)

        class_module = getattr(p, n).__module__
        assert class_module.startswith("daal4py") or class_module.startswith(
            "sklearnex"
        ), "Patching has completed with error."

    sklearnex.unpatch_sklearn()


def test_patch_by_list_simple():
    sklearnex.patch_sklearn(["LogisticRegression"])

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVC

    assert RandomForestRegressor.__module__.startswith("sklearn")
    assert KNeighborsRegressor.__module__.startswith("sklearn")
    assert LogisticRegression.__module__.startswith("daal4py")
    assert SVC.__module__.startswith("sklearn")

    sklearnex.unpatch_sklearn()


def test_patch_by_list_many_estimators():
    sklearnex.patch_sklearn(["LogisticRegression", "SVC"])

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVC

    assert RandomForestRegressor.__module__.startswith("sklearn")
    assert KNeighborsRegressor.__module__.startswith("sklearn")
    assert LogisticRegression.__module__.startswith("daal4py")
    assert SVC.__module__.startswith("daal4py") or SVC.__module__.startswith("sklearnex")

    sklearnex.unpatch_sklearn()


def test_unpatch_by_list_many_estimators():
    sklearnex.patch_sklearn()

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVC

    assert RandomForestRegressor.__module__.startswith("sklearnex")
    assert KNeighborsRegressor.__module__.startswith(
        "daal4py"
    ) or KNeighborsRegressor.__module__.startswith("sklearnex")
    assert LogisticRegression.__module__.startswith("daal4py")
    assert SVC.__module__.startswith("daal4py") or SVC.__module__.startswith("sklearnex")

    sklearnex.unpatch_sklearn(["KNeighborsRegressor", "RandomForestRegressor"])

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVC

    assert RandomForestRegressor.__module__.startswith("sklearn")
    assert KNeighborsRegressor.__module__.startswith("sklearn")
    assert LogisticRegression.__module__.startswith("daal4py")
    assert SVC.__module__.startswith("daal4py") or SVC.__module__.startswith("sklearnex")


def test_patching_checker():
    for name in [None, "SVC", "PCA"]:
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


def test_preview_namespace():
    def get_estimators():
        from sklearn.cluster import DBSCAN
        from sklearn.decomposition import PCA
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LinearRegression
        from sklearn.svm import SVC

        return LinearRegression(), PCA(), DBSCAN(), SVC(), RandomForestClassifier()

    # BUG: previous patching tests force PCA to be patched with daal4py.
    # This unpatching returns behavior to expected
    sklearnex.unpatch_sklearn()
    # behavior with enabled preview
    sklearnex.patch_sklearn(preview=True)
    assert sklearnex.dispatcher._is_preview_enabled()

    lr, pca, dbscan, svc, rfc = get_estimators()
    assert "sklearnex" in rfc.__module__

    if daal_check_version((2023, "P", 100)):
        assert "sklearnex" in lr.__module__
    else:
        assert "daal4py" in lr.__module__

    assert "sklearnex.preview" in pca.__module__
    assert "sklearnex" in dbscan.__module__
    assert "sklearnex" in svc.__module__
    sklearnex.unpatch_sklearn()

    # no patching behavior
    lr, pca, dbscan, svc, rfc = get_estimators()
    assert "sklearn." in lr.__module__
    assert "sklearn." in pca.__module__
    assert "sklearn." in dbscan.__module__
    assert "sklearn." in svc.__module__
    assert "sklearn." in rfc.__module__

    # default patching behavior
    sklearnex.patch_sklearn()
    assert not sklearnex.dispatcher._is_preview_enabled()

    lr, pca, dbscan, svc, rfc = get_estimators()
    if daal_check_version((2023, "P", 100)):
        assert "sklearnex" in lr.__module__
    else:
        assert "daal4py" in lr.__module__
    assert "daal4py" in pca.__module__
    assert "sklearnex" in rfc.__module__
    assert "sklearnex" in dbscan.__module__
    assert "sklearnex" in svc.__module__
    sklearnex.unpatch_sklearn()
