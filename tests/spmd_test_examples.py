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

from test_examples import np_read_csv, add_test


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
