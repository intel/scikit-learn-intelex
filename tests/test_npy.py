import unittest
import numpy as np
from daal4py import __daal_link_version__ as dv, linear_regression_prediction, linear_regression_training
daal_version = tuple(map(int, (dv[0:4], dv[4:8])))


class Test(unittest.TestCase):
    def test_non_contig(self):
        from numpy.random import rand
        p = 10007
        nx = 1017
        ny = 77
        X = rand(p+1,nx+1)
        Xp = rand(p+1,nx+1)
        y = rand(p+1,ny+1)
        Xn = X[:p,:nx]
        Xpn = Xp[:p,:nx]
        yn = y[:p,:ny]
        Xc = np.ascontiguousarray(Xn)
        Xpc = np.ascontiguousarray(Xpn)
        yc = np.ascontiguousarray(yn)
        self.assertTrue(not Xn.flags['C_CONTIGUOUS'] and not Xpn.flags['C_CONTIGUOUS'] and not yn.flags['C_CONTIGUOUS'])
        self.assertTrue(Xc.flags['C_CONTIGUOUS'] and Xpc.flags['C_CONTIGUOUS'] and yc.flags['C_CONTIGUOUS'])
        self.assertTrue(np.allclose(Xc, Xn) and np.allclose(Xpc, Xpn) and np.allclose(yc, yn))
        regr_train = linear_regression_training()
        rtc = regr_train.compute(Xc, yc)
        regr_predict = linear_regression_prediction()
        rpc = regr_predict.compute(Xpc, rtc.model)
        regr_train = linear_regression_training()
        rtn = regr_train.compute(Xn, yn)
        regr_predict = linear_regression_prediction()
        rpn = regr_predict.compute(Xpn, rtn.model)
        self.assertTrue(np.allclose(rpn.prediction, rpc.prediction))

if __name__ == '__main__':
    unittest.main()
