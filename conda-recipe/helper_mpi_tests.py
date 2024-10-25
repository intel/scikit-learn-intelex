# The purpose of this helper script is to be able to execute
# different shell commands according to whether the process
# is the first MPI rank or not. It works by passing it two
# arguments:
# - First argument is what gets executed by ranks other than '0'.
# - Second argument is what gets executed by rank 0.
# Note that the arguments might be multi-word comments, so it
# uses os.system instead of subprocess or similar.

import os
import sys

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    is_rank_zero = rank == 0
except ImportError:
    is_rank_zero = True

if is_rank_zero:
    os.system(sys.argv[-1])
else:
    os.system(sys.argv[-2])
