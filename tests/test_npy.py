import unittest
import numpy as np
import daal4py as d4p

dv = d4p.__daal_link_version__
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
        regr_train = d4p.linear_regression_training()
        rtc = regr_train.compute(Xc, yc)
        regr_predict = d4p.linear_regression_prediction()
        rpc = regr_predict.compute(Xpc, rtc.model)
        regr_train = d4p.linear_regression_training()
        rtn = regr_train.compute(Xn, yn)
        regr_predict = d4p.linear_regression_prediction()
        rpn = regr_predict.compute(Xpn, rtn.model)
        self.assertTrue(np.allclose(rpn.prediction, rpc.prediction))

    def test_struct(self):
        sdata = np.array([(0.5, -1.3, 1, 100.11, 1111111),
                           (2.5, -3.3, 2, 200.22, 2222222),
                           (4.5, -5.3, 2, 350.33, 3333333),
                           (6.5, -7.3, 0, 470.44, 4444444),
                           (8.5, -9.3, 1, 270.55, 55555)],
                          dtype=[('x','f4'), ('y','f4'), ('categ','i4'), ('value','f8'), ('super','i8')])
        hdata = np.array([(0.5, -1.3, 1, 100.11, 1111111),
                           (2.5, -3.3, 2, 200.22, 2222222),
                           (4.5, -5.3, 2, 350.33, 3333333),
                           (6.5, -7.3, 0, 470.44, 4444444),
                           (8.5, -9.3, 1, 270.55, 55555)],
                          dtype = np.float64)
        sr = d4p.cosine_distance().compute(sdata)
        hr = d4p.cosine_distance().compute(hdata)
        self.assertTrue(np.allclose(hr.cosineDistance, sr.cosineDistance))


if __name__ == '__main__':
    unittest.main()
