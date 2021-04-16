import argparse
import pytest
import os


def get_context(device):
    from daal4py.oneapi import sycl_context
    return sycl_context(device, host_offload_on_fail=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script to run scikit-learn tests with device context manager')
    parser.add_argument(
        '-q', '--quiet',
        help='make pytest less verbose',
        action='store_false')
    parser.add_argument(
        '-d', '--device',
        type=str,
        help='device name',
        choices=['host', 'cpu', 'gpu']
    )
    parser.add_argument(
        '--deselect',
        help='The list of deselect commands passed directly to pytest',
        action='append',
        required=True
    )
    args = parser.parse_args()

    deselected_tests = [
        element for test in args.deselect
        for element in ('--deselect', test)
    ]

    pytest_params = [
        "-ra", "--disable-warnings"
    ]

    if not args.quiet:
        pytest_params.append("-q")

    with get_context(args.device):
        pytest.main(
            pytest_params + ["--pyargs", "sklearn"] + deselected_tests
        )
