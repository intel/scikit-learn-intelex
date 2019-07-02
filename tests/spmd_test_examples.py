import daal4py as d4p

if d4p.__has_dist__:
    import unittest
    import numpy as np
    from test_examples import np_read_csv, add_test


    class Base():
        def test_svd_spmd(self):
            import svd_spmd as ex
            (data, result) = self.call(ex)
            self.assertTrue(np.allclose(data, np.matmul(np.matmul(result.leftSingularMatrix, np.diag(result.singularValues[0])), result.rightSingularMatrix)))

        def test_qr_spmd(self):
            import qr_spmd as ex
            (data, result) = self.call(ex)
            self.assertTrue(np.allclose(data, np.matmul(result.matrixQ, result.matrixR)))

        def test_kmeans_spmd(self):
            import kmeans_spmd as ex
            (assignments, result) = self.call(ex)
            data = "./data/distributed/kmeans_dense.csv"
            nClusters = 10
            maxIter = 25
            batch_init_res = d4p.kmeans_init(nClusters=nClusters, method="plusPlusDense").compute(data)
            batch_result = d4p.kmeans(nClusters=nClusters, maxIterations=maxIter, assignFlag=True).compute(data, batch_init_res.centroids)
            self.assertTrue(np.allclose(result.centroids, batch_result.centroids))


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
