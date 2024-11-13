import pytest
from sklearnex._utils import StaticOnlyMethod


def test_staticonlymethod_default_args():
    class Test:
        @StaticOnlyMethod
        def test_static_method():
            pass

    with pytest.raises(AttributeError):
        Test().test_static_method()

    assert Test.test_static_method() is None


def test_staticonlymethod_custom_exception():
    """Can raise a custom exception when called on an instance."""

    class Test:
        @StaticOnlyMethod(instance_call_behavior=ValueError("Custom error"))
        def test_static_method():
            pass

    with pytest.raises(ValueError):
        Test().test_static_method()

    assert Test.test_static_method() is None


def test_staticonlymethod_custom_warning():
    """Can raise a custom warning when called on an instance."""

    class Test:
        @StaticOnlyMethod(instance_call_behavior=Warning("Custom warning"))
        def test_static_method():
            pass

    with pytest.warns(Warning):
        Test().test_static_method()

    assert Test.test_static_method() is None
