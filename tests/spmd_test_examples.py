import os
import sys
test_path = os.path.abspath(os.path.dirname(__file__))
unittest_data_path = os.path.join(test_path, "unittest_data")
examples_path = os.path.join(os.path.dirname(test_path), "examples")
sys.path.insert(0, examples_path)
os.chdir(examples_path)

import unittest
import daal4py as d4p
import numpy as np

from daal4py import __daal_link_version__ as dv
daal_version = tuple(map(int, (dv[0:4], dv[4:8])))

# function reading file and returning numpy array
def np_read_csv(f, c=None, s=0, n=np.iinfo(np.int64).max, t=np.float64):
    if s==0 and n==np.iinfo(np.int64).max:
        return np.loadtxt(f, usecols=c, delimiter=',', ndmin=2, dtype=t)
    a = np.genfromtxt(f, usecols=c, delimiter=',', skip_header=s, max_rows=n, dtype=t)
    if a.shape[0] == 0:
        raise Exception("done")
    if a.ndim == 1:
        return a[:, np.newaxis]
    return a


def add_test(cls, e, f=None, attr=None, ver=(0,0)):
    import importlib
    @unittest.skipIf(daal_version < ver, "not supported in this library version")
    def testit(self):
        ex = importlib.import_module(e)
        result = self.call(ex)
        if f and attr:
            testdata = np_read_csv(os.path.join(unittest_data_path, f))
            actual = attr(result) if callable(attr) else getattr(result, attr)
            self.assertTrue(np.allclose(actual, testdata, atol=1e-05), msg="Discrepancy found: {}".format(np.abs(actual-testdata).max()))
        else:
            self.assertTrue(True)
    setattr(cls, 'test_'+e, testit)


class Base():
    pass


gen_examples = [
    ('covariance_spmd', 'covariance.csv', 'covariance'),
   ('low_order_moms_spmd', 'low_order_moms_dense_batch.csv', lambda r: np.vstack((r.minimum,
                                                                                          r.maximum,
                                                                                          r.sum,
                                                                                          r.sumSquares,
                                                                                          r.sumSquaresCentered,
                                                                                          r.mean,
                                                                                          r.secondOrderRawMoment,
                                                                                          r.variance,
                                                                                          r.standardDeviation,
                                                                                          r.variation))),
]


for example in gen_examples:
    add_test(Base, *example)


class Test(Base, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d4p.daalinit()

    @classmethod
    def tearDownClass(cls):
        d4p.daalfini()

    def call(self, ex):
        return ex.main()
        

if __name__ == '__main__':
    unittest.main()
