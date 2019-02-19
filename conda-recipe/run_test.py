import os
import sys

# Work around Intel(R) MPI package issue with libfabric
if sys.platform in ['win32', 'cygwin']:
    os.environ['PATH'] = ';'.join([os.environ['PATH'], os.path.join(os.environ['CONDA_PREFIX'], 'Library', 'bin', 'libfabric')])

here = os.path.abspath(os.path.dirname(__file__))
ex_dir = os.path.join(here, "examples")

from examples.run_examples import run_all

os.chdir(ex_dir)
ret = run_all()

sys.exit(4711 if ret else 0)
