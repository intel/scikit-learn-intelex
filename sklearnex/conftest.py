import io
import logging

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "allow_sklearn_fallback: mark test to not check for sklearnex usage"
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    # setup logger to check for sklearn fallback
    if not item.get_closest_marker("allow_sklearn_fallback"):

        log_stream = io.StringIO()
        log_handler = logging.StreamHandler(log_stream)
        sklearnex_logger = logging.getLogger("sklearnex")
        level = sklearnex_logger.level
        sklearnex_logger.addHandler(log_handler)
        sklearnex_logger.setLevel(logging.INFO)

        yield

        log_handler.setLevel(level)
        sklearnex_logger.removeHandler(log_handler)

        if "fallback to original Scikit-learn" in log_stream.getvalue():
            raise TypeError(
                "sklearnex test evaluated with sklearn via fallback mechanism"
            )
    else:
        yield
