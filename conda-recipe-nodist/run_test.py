import sys
import unittest
from tests.test_examples import Test
s = unittest.defaultTestLoader.discover('tests')
r = unittest.TextTestRunner()
r.run(s)
sys.exit(0 if r._makeResult().wasSuccessful() else 1)
