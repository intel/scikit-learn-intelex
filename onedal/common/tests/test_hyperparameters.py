import pytest
from sklearnex._utils import register_hyperparameters


def test_register_hyperparameters():
    hyperparameters_map = {"op": "hyperparameters"}

    @register_hyperparameters(hyperparameters_map)
    class Test:
        pass

    # assert the correct value is returned
    assert Test.get_hyperparameters("op") == "hyperparameters"
    # assert a warning is issued when trying to modify the hyperparameters per instance
    with pytest.warns(Warning):
        Test().get_hyperparameters("op")
