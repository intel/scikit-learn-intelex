import os
import sys
import unittest

if sys.platform in ['win32', 'cygwin']:
    os.environ['PATH'] = ';'.join([os.environ['PATH'], os.path.join(os.environ['CONDA_PREFIX'], 'Library', 'bin', 'libfabric')])

here = os.path.abspath(os.path.dirname(__file__))
ex_dir = os.path.join(here, "examples")

from examples.run_examples import run_all
from tests.test_examples import Test

s = unittest.defaultTestLoader.discover('tests')
r = unittest.TextTestRunner()
r.run(s)
ret1 = 0 if r._makeResult().wasSuccessful() else 1

os.chdir(ex_dir)
ret2 = run_all(True)

sys.exit(ret1 + ret2)
