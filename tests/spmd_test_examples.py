import daal4py as d4p
import os

if d4p.__has_dist__:
    import unittest
    import numpy as np
    from test_examples import np_read_csv, add_test, daal_version, unittest_data_path


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

        @unittest.skipIf(daal_version < (2019, 5), "not supported in this library version")
        def test_dbscan_spmd(self):
            import dbscan_spmd as ex
            result = self.call(ex)
            test_data = np_read_csv(os.path.join(unittest_data_path, "dbscan_batch.csv"))
            rpp = int(test_data.shape[0]/d4p.num_procs())
            test_data = test_data[rpp*d4p.my_procid():rpp*d4p.my_procid()+rpp,:]
            # clusters can get different indexes in batch and spmd algos, to compare them we should take care about it
            cluster_index_dict = {}
            for i in range(test_data.shape[0]):
                if not test_data[i][0] in cluster_index_dict:
                    cluster_index_dict[test_data[i][0]] = result.assignments[i][0]
                self.assertTrue(cluster_index_dict[test_data[i][0]] == result.assignments[i][0])


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
