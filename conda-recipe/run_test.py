import os
import sys
import unittest

here = os.path.abspath(os.path.dirname(__file__))
ex_dir = os.path.join(os.path.dirname(here), "examples")

from examples.run_examples import run_all
from tests.test_examples import Test

s = unittest.defaultTestLoader.discover('tests')
r = unittest.TextTestRunner()
r.run(s)
ret1 = 0 if r._makeResult().wasSuccessful() else 1

os.chdir(ex_dir)
ret2 = run_all()

sys.exit(ret1 + ret2)
