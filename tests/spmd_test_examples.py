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
            nClusters = 10
            maxIter = 25

            data = np.loadtxt("./data/distributed/kmeans_dense.csv", delimiter=',')

            rpp = int(data.shape[0]/d4p.num_procs())
            spmd_data = data[rpp*d4p.my_procid():rpp*d4p.my_procid()+rpp,:]

            for init_method in ['plusPlusDense', 'parallelPlusDense', 'deterministicDense']:
                batch_init_res = d4p.kmeans_init(nClusters=nClusters, method=init_method).compute(data)
                spmd_init_res = d4p.kmeans_init(nClusters=nClusters, method=init_method, distributed=True).compute(spmd_data)

                if init_method in ['parallelPlusDense']:
                    print("Warning: It is well known that results of parallelPlusDense init does not match with batch algorithm")
                else:
                    self.assertTrue(np.allclose(batch_init_res.centroids, spmd_init_res.centroids),
                                    "Initial centroids with " + init_method + " does not match with batch algorithm")

                batch_res = d4p.kmeans(nClusters=nClusters, maxIterations=maxIter).compute(data, batch_init_res.centroids)
                spmd_res = d4p.kmeans(nClusters=nClusters, maxIterations=maxIter, distributed=True).compute(spmd_data, spmd_init_res.centroids)

                self.assertTrue(np.allclose(batch_res.centroids, batch_res.centroids),
                                "Final centroids with " + init_method + " does not match with batch algorithm")


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
