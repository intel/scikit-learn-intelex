import pytest

from daal4py.sklearn import _utils


def injected_get_daal_version():
    return (2020, "P", 100)


@pytest.mark.parametrize(
    "required_version,expected",
    [
        ((2020, "P", 100), True),
        ((2019, "P", 100), True),
        ((2020, "A", 100), True),
        ((2020, "P", 1), True),
        ((2021, "P", 100), False),
        ((2020, "Z", 100), False),
        ((2020, "P", 200), False),
        ((2019, "A", 1), True),
        ((2021, "Z", 200), False),
        ((2019, "P", 200), True),
        (((2020, "P", 100), (2021, "Z", 200)), True),
        (((2019, "P", 100), (2018, "P", 100)), True),
        (((2019, "P", 100), (2018, "P", 100)), True),
        (((2021, "P", 100), (2022, "P", 100)), False),
    ],
)
def test_daal_check_version(required_version, expected):
    actual = _utils.daal_check_version(required_version, injected_get_daal_version)
    assert actual == expected, f"{required_version=}, {expected=}, {actual=}"
