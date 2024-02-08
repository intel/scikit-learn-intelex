import pytest

from sklearnex import patch_sklearn, unpatch_sklearn


@pytest.fixture
def with_sklearnex():
    patch_sklearn()
    yield
    unpatch_sklearn()
