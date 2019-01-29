import os
import sys
import unittest

# Work around Intel(R) MPI package issue with libfabric
if sys.platform in ['win32', 'cygwin']:
    os.environ['PATH'] = ';'.join([os.environ['PATH'], os.path.join(os.environ['CONDA_PREFIX'], 'Library', 'bin', 'libfabric')])

here = os.path.abspath(os.path.dirname(__file__))
ex_dir = os.path.join(here, "examples")

from examples.run_examples import run_all
#from tests.test_examples import TestExNpy, TestExPd

s = unittest.defaultTestLoader.discover('tests')
r = unittest.TextTestRunner(verbosity=7)

testResult = r.run(s)
ret1 = 0 if testResult.wasSuccessful() else 1

os.chdir(ex_dir)
ret2 = run_all()

sys.exit(4711 if ret1 or ret2 else 0)
