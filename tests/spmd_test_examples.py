import unittest
import daal4py as d4p
import numpy as np

from test_examples import np_read_csv, add_test


class Base():
    def test_svd_spmd(self):
        import svd_spmd as ex
        (data, result) = self.call(ex)
        self.assertTrue(np.allclose(data, np.matmul(np.matmul(result.leftSingularMatrix,np.diag(result.singularValues[0])),result.rightSingularMatrix)))


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
