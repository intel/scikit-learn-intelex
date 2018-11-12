import sys
import unittest
from tests.test_examples import Test
s = unittest.defaultTestLoader.discover('tests')
r = unittest.TextTestRunner()
r.run(s)
ret1 = 0 if r._makeResult().wasSuccessful() else 1

from examples.run_examples import run_all
ret2 = run_all()

sys.exit(ret1 + ret2)
