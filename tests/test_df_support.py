import unittest
import numpy as np
import pandas as pd
import daal4py as d4p


class Test(unittest.TestCase):
    def verify_on_dbscan(self, X):
        alg1 = d4p.dbscan(epsilon=2.0, minObservations=5, fptype='double')
        res1 = alg1.compute(X)
        Xc = np.ascontiguousarray(X)
        alg2 = d4p.dbscan(epsilon=2.0, minObservations=5, fptype='double')
        res2 = alg2.compute(Xc)
        self.assertTrue(np.array_equal(res1.assignments, res2.assignments))
        self.assertTrue(len(np.unique(res1.assignments)) > 2)

    def verify_on_linear_regression(self, X, Y):
        alg1 = d4p.linear_regression_training(interceptFlag=True, fptype='double')
        res1 = alg1.compute(X, Y)
        Xc = np.ascontiguousarray(X)
        Yc = np.ascontiguousarray(Y).reshape((len(Y), 1))
        alg2 = d4p.linear_regression_training(interceptFlag=True, fptype='double')
        res2 = alg2.compute(Xc, Yc)
        print(res1.model.Beta[:10])
        print(res2.model.Beta[:10])
        self.assertTrue(np.allclose(res1.model.Beta, res2.model.Beta))


    def test1(self):
        """
        Dataframe from C-contiguous array
        """
        X = np.random.randn(13024, 16)
        df = pd.DataFrame(X)
        self.verify_on_dbscan(df)

    def test2(self):
        """
        Dataframe from F-contiguous array
        """
        X = np.asfortranarray(np.random.randn(13024, 16))
        df = pd.DataFrame(X)
        self.verify_on_dbscan(df)

    def test3(self):
        """
        Dataframe from non-contiguous array, case 1
        """
        X = np.random.randn(13024, 5, 16)
        df = pd.DataFrame(X[:, 2, :])
        self.verify_on_dbscan(df)


    def test4(self):
        """
        Dataframe from non-contiguous array, case 2
        """
        X = np.random.randn(13024*3, 16)
        df = pd.DataFrame(X[1::3, :])
        self.verify_on_dbscan(df)

    def test5(self):
        """
        Dataframe and Series from non-contiguous array, case 3
        """
        X = np.random.randn(13024*3, 16)
        df = pd.DataFrame(X[1::3, 1:])
        ps = pd.Series(X[1::3, 0])
        print(ps.to_numpy().flags)
        self.verify_on_linear_regression(df, ps)

    def test6(self):
        """
        Dataframe from C-contiguous array with heterogeneous types
        """
        X = np.random.randn(13024, 16)
        df = pd.DataFrame(X)
        df = df.astype({df.columns[1]: 'float32', df.columns[6]: 'float32'})
        self.verify_on_dbscan(df)

    def test7(self):
        """
        Dataframe from non-contiguous array with heterogeneous types
        """
        X = np.random.randn(13024*3, 16)
        df = pd.DataFrame(X[1::3, :])
        df = df.astype({df.columns[1]: 'float32', df.columns[6]: 'float32'})
        self.verify_on_dbscan(df)

    def test8(self):
        """
        Dataframe from multi-dtype array
        """
        X = np.random.randn(13024, 16)
        dt = []
        for i in range(8):
            dt += [('f' + str(2*i), np.float64),
                   ('f' + str(2*i+1), np.float32)]
        X2 = np.empty((X.shape[0],), dtype=np.dtype(dt))
        for i, (n, _) in enumerate(dt):
            X2[n][:] = X[:, i]

        df = pd.DataFrame(X2)
        for i, (n, _) in enumerate(dt):
            assert np.allclose(df[n].values, X[:, i])
        self.verify_on_dbscan(df)

    def test9(self):
        """
        Check own data for to_numpy method
        """
        X = np.random.randn(13024*3, 16)
        df = pd.DataFrame(X[1::3, :])
        print(df.to_numpy().data, X.data)
        self.assertTrue(np.may_share_memory(df.to_numpy(), X))


if __name__ == '__main__':
    unittest.main()
