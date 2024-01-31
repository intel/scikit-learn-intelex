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
        sklearnex_stderr_handler = sklearnex_logger.handlers
        sklearnex_logger.handlers = []
        sklearnex_logger.addHandler(log_handler)
        sklearnex_logger.setLevel(logging.INFO)
        log_handler.setLevel(logging.INFO)

        yield

        sklearnex_logger.handlers = sklearnex_stderr_handler
        sklearnex_logger.setLevel(level)
        sklearnex_logger.removeHandler(log_handler)
        text = log_stream.getvalue()
        if "fallback to original Scikit-learn" in text:
            raise TypeError(
                f"test did not properly evaluate sklearnex functionality and fell back to sklearn:\n{text}"
            )
    else:
        yield
