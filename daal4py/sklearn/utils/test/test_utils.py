import pytest

from daal4py.sklearn import _utils


def injected_get_daal_version():
    return (2020, "C", 100)


@pytest.mark.parametrize(
    "required_version,expected",
    [
        ((2020, "C", 100), True),
        ((2019, "C", 100), True),
        ((2020, "B", 100), True),
        ((2020, "C", 99), True),
        ((2021, "C", 100), False),
        ((2020, "D", 100), False),
        ((2020, "C", 101), False),
        ((2019, "B", 99), True),
        ((2021, "D", 101), False),
        (((2020, "C", 100), (2021, "D", 101)), True),
        (((2019, "C", 100), (2018, "C", 100)), True),
        (((2019, "C", 100), (2018, "C", 100)), True),
        (((2021, "C", 100), (2022, "C", 100)), False),
    ],
)
def test_daal_check_version(required_version, expected):
    actual = _utils.daal_check_version(required_version, injected_get_daal_version)
    assert actual == expected, f"{required_version=}, {expected=}, {actual=}"
