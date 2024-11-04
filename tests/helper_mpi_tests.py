# The purpose of this helper script is to be able to execute
# different shell commands according to whether the process
# is the first MPI rank or not. It works by passing it the
# arguments to a pytest call with the json report arguments
# from an mpirun/mpiexec invocation - e.g.
#     mpirun -n 2 helper_mpi_tests.py pytest <args to pytest>

import sys

import pytest

print("Info from 'helper_mpi_tests.py'")
print("sys.argv:", sys.argv)

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    is_rank_zero = rank == 0
except ImportError:
    is_rank_zero = True

args_base = [
    arg for arg in sys.argv[1:] if not ("--json-report" in arg) and (arg != "pytest")
]
args_json_report = [
    arg for arg in sys.argv[1:] if ("--json-report" in arg) and (arg != "pytest")
]

if is_rank_zero:
    pytest.main(args_base + args_json_report)
else:
    pytest.main(args_base)
